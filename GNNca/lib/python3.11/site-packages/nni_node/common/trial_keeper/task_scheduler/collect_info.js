"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.collectGpuInfo = void 0;
const log_1 = require("common/log");
const pythonScript_1 = require("common/pythonScript");
const logger = (0, log_1.getLogger)('GpuInfoCollector');
let cache = null;
const minUpdateInterval = 10 * 1000;
async function collectGpuInfo(forceUpdate) {
    if (!forceUpdate && cache !== null) {
        if (Date.now() - cache.timestamp < minUpdateInterval) {
            return cache;
        }
    }
    let str;
    try {
        const args = (forceUpdate ? ['--detail'] : undefined);
        str = await (0, pythonScript_1.runPythonModule)('nni.tools.nni_manager_scripts.collect_gpu_info', args);
    }
    catch (error) {
        logger.error('Failed to collect GPU info:', error);
        return null;
    }
    let info;
    try {
        info = JSON.parse(str);
    }
    catch (error) {
        logger.error('Failed to collect GPU info, collector output:', str);
        return null;
    }
    if (!info.success) {
        logger.error('Failed to collect GPU info, collector output:', info);
        return null;
    }
    if (forceUpdate) {
        logger.info('Forced update:', info);
    }
    else {
        logger.debug(info);
    }
    cache = info;
    return info;
}
exports.collectGpuInfo = collectGpuInfo;
