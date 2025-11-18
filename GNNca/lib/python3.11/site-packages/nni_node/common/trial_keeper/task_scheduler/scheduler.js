"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.TaskScheduler = void 0;
const node_events_1 = require("node:events");
const log_1 = require("common/log");
const collect_info_1 = require("./collect_info");
const logger = (0, log_1.getLogger)('TaskScheduler');
let collectGpuInfo = collect_info_1.collectGpuInfo;
class TaskScheduler {
    emitter = new node_events_1.EventEmitter();
    gpus = [];
    trials = [];
    async init() {
        const info = await collectGpuInfo(true);
        if (info === null) {
            throw new Error('TaskScheduler: Failed to collect GPU info');
        }
        if (info.gpuNumber === 0) {
            throw new Error('TaskScheduler: No GPU found');
        }
        for (let i = 0; i < info.gpuNumber; i++) {
            this.gpus.push({
                index: i,
                util: 0,
                coreUtil: 0,
                memUtil: 0,
                active: false,
                computeActive: false,
            });
        }
        this.updateGpus(info);
    }
    async update(force) {
        const info = await collectGpuInfo(force);
        if (info === null) {
            if (force) {
                throw new Error('TaskScheduler: Failed to update GPU info');
            }
            return;
        }
        if (info.gpuNumber !== this.gpus.length) {
            throw new Error(`TaskScheduler: GPU number changed from ${this.gpus.length} to ${info.gpuNumber}`);
        }
        this.updateGpus(info);
    }
    async schedule(experimentId, trialId, gpuNumber, restrictions) {
        if (gpuNumber === 0) {
            return { 'CUDA_VISIBLE_DEVICES': '' };
        }
        this.update();
        if (gpuNumber >= this.gpus.length) {
            logger.error(`Only have ${this.gpus.length} GPUs, requesting ${gpuNumber}`);
            return null;
        }
        const gpus = this.sortGpus(restrictions ?? {});
        if (gpuNumber < 1) {
            const gpu = gpus[0];
            if (gpu.util + gpuNumber > 1.001) {
                return null;
            }
            gpu.util += gpuNumber;
            this.trials.push({ gpuIndex: gpu.index, experimentId, trialId, util: gpuNumber });
            logger.debug(`Scheduled ${experimentId}/${trialId} -> ${gpu.index}`);
            this.emitUpdate();
            return { 'CUDA_VISIBLE_DEVICES': String(gpu.index) };
        }
        else {
            const n = Math.round(gpuNumber);
            if (gpus.length < n || gpus[n - 1].util > 0) {
                return null;
            }
            const indices = [];
            for (const gpu of gpus.slice(0, n)) {
                gpu.util = 1;
                this.trials.push({ gpuIndex: gpu.index, experimentId, trialId, util: 1 });
                indices.push(gpu.index);
            }
            indices.sort((a, b) => (a - b));
            logger.debug(`Scheduled ${experimentId}/${trialId} ->`, indices);
            return { 'CUDA_VISIBLE_DEVICES': indices.join(',') };
        }
    }
    async release(experimentId, trialId) {
        this.releaseByFilter(trial => (trial.experimentId === experimentId && trial.trialId === trialId));
    }
    async releaseAll(experimentId) {
        logger.info('Release whole experiment', experimentId);
        this.releaseByFilter(trial => (trial.experimentId === experimentId));
    }
    onUtilityUpdate(callback) {
        this.emitter.on('update', callback);
    }
    updateGpus(info) {
        const prev = structuredClone(this.gpus);
        for (const gpu of info.gpus) {
            const index = gpu.index;
            this.gpus[index].coreUtil = gpu.gpuCoreUtilization ?? 0;
            this.gpus[index].memUtil = gpu.gpuMemoryUtilization ?? 0;
            this.gpus[index].active = false;
            this.gpus[index].computeActive = false;
        }
        for (const proc of info.processes) {
            const index = proc.gpuIndex;
            this.gpus[index].active = true;
            if (proc.type === 'compute') {
                this.gpus[index].computeActive = true;
            }
        }
        for (let i = 0; i < this.gpus.length; i++) {
            const prevUtil = Math.max(prev[i].util, prev[i].coreUtil, prev[i].memUtil);
            const curUtil = Math.max(this.gpus[i].util, this.gpus[i].coreUtil, this.gpus[i].memUtil);
            if (Math.abs(prevUtil - curUtil) > 0.5) {
                this.emitUpdate(info);
                return;
            }
            const prevActive = prev[i].util > 0 || prev[i].active;
            const curActive = this.gpus[i].util > 0 || this.gpus[i].active;
            if (prevActive !== curActive) {
                this.emitUpdate(info);
                break;
            }
        }
    }
    sortGpus(restrict) {
        let gpus = this.gpus.slice();
        if (restrict.onlyUseIndices) {
            gpus = gpus.filter(gpu => restrict.onlyUseIndices.includes(gpu.index));
        }
        if (restrict.rejectActive) {
            gpus = gpus.filter(gpu => !gpu.active);
        }
        if (restrict.rejectComputeActive) {
            gpus = gpus.filter(gpu => !gpu.computeActive);
        }
        return gpus.sort((a, b) => {
            if (a.util !== b.util) {
                return a.util - b.util;
            }
            if (a.active !== b.active) {
                return Number(a.active) - Number(b.active);
            }
            if (a.computeActive !== b.computeActive) {
                return Number(a.computeActive) - Number(b.computeActive);
            }
            if (a.memUtil !== b.memUtil) {
                return a.memUtil - b.memUtil;
            }
            if (a.coreUtil !== b.coreUtil) {
                return a.coreUtil - b.coreUtil;
            }
            return a.index - b.index;
        });
    }
    releaseByFilter(filter) {
        const trials = this.trials.filter(filter);
        trials.forEach(trial => {
            logger.debug(`Released ${trial.experimentId}/${trial.trialId}`);
            this.gpus[trial.gpuIndex].util -= trial.util;
        });
        this.trials = this.trials.filter(trial => !filter(trial));
        if (trials) {
            this.emitUpdate();
        }
    }
    async emitUpdate(info) {
        const copy = structuredClone(info ?? await collectGpuInfo());
        if (copy) {
            for (const gpu of copy.gpus) {
                gpu.nomialUtilization = this.gpus[gpu.index].util;
            }
            this.emitter.emit('update', { gpu: copy });
        }
    }
}
exports.TaskScheduler = TaskScheduler;
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function mockGpuInfo(info) {
        collectGpuInfo = (_) => Promise.resolve(info);
    }
    UnitTestHelpers.mockGpuInfo = mockGpuInfo;
    function getGpuUtils(scheduler) {
        return scheduler.gpus.map((gpu) => gpu.util);
    }
    UnitTestHelpers.getGpuUtils = getGpuUtils;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
