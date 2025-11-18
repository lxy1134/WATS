"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DlcEnvironmentService = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const ts_deferred_1 = require("ts-deferred");
const ioc_shim_1 = require("common/ioc_shim");
const log_1 = require("common/log");
const dlcClient_1 = require("../dlc/dlcClient");
const dlcConfig_1 = require("../dlc/dlcConfig");
const environment_1 = require("../environment");
const fileCommandChannel_1 = require("../channels/fileCommandChannel");
const mountedStorageService_1 = require("../storages/mountedStorageService");
const storageService_1 = require("../storageService");
const utils_1 = require("common/utils");
const promises_1 = require("timers/promises");
class DlcEnvironmentService extends environment_1.EnvironmentService {
    log = (0, log_1.getLogger)('dlcEnvironmentService');
    experimentId;
    config;
    constructor(config, info) {
        super();
        this.experimentId = info.experimentId;
        this.config = config;
        ioc_shim_1.IocShim.bind(storageService_1.StorageService, mountedStorageService_1.MountedStorageService);
        const storageService = ioc_shim_1.IocShim.get(storageService_1.StorageService);
        const remoteRoot = storageService.joinPath(this.config.localStorageMountPoint, 'nni-experiments', this.experimentId);
        const localRoot = storageService.joinPath(this.config.localStorageMountPoint, 'nni-experiments');
        storageService.initialize(localRoot, remoteRoot);
    }
    get hasStorageService() {
        return true;
    }
    initCommandChannel(eventEmitter) {
        this.commandChannel = new fileCommandChannel_1.FileCommandChannel(eventEmitter);
    }
    createEnvironmentInformation(envId, envName) {
        return new dlcConfig_1.DlcEnvironmentInformation(envId, envName);
    }
    get getName() {
        return 'dlc';
    }
    async refreshEnvironmentsStatus(environments) {
        const deferred = new ts_deferred_1.Deferred();
        environments.forEach(async (environment) => {
            const dlcClient = environment.dlcClient;
            if (!dlcClient) {
                return Promise.reject('DLC client not initialized!');
            }
            const newStatus = await dlcClient.updateStatus(environment.status);
            switch (newStatus.toUpperCase()) {
                case 'CREATING':
                case 'CREATED':
                case 'WAITING':
                case 'QUEUED':
                    environment.setStatus('WAITING');
                    break;
                case 'RUNNING':
                    environment.setStatus('RUNNING');
                    break;
                case 'COMPLETED':
                case 'SUCCEEDED':
                    environment.setStatus('SUCCEEDED');
                    break;
                case 'FAILED':
                    await (0, promises_1.setTimeout)(60000);
                    this.log.debug(`await 60s to create new job,DLC: job ${environment.id} is failed!`);
                    environment.setStatus('FAILED');
                    break;
                case 'STOPPED':
                case 'STOPPING':
                    environment.setStatus('USER_CANCELED');
                    break;
                default:
                    environment.setStatus('UNKNOWN');
            }
        });
        deferred.resolve();
        return deferred.promise;
    }
    async startEnvironment(environment) {
        const dlcEnvironment = environment;
        const environmentRoot = path_1.default.join(this.config.containerStorageMountPoint, `/nni-experiments/${this.experimentId}`);
        const localRoot = path_1.default.join(this.config.localStorageMountPoint, `/nni-experiments/${this.experimentId}`);
        dlcEnvironment.workingFolder = `${localRoot}/envs/${environment.id}`;
        dlcEnvironment.runnerWorkingFolder = `${environmentRoot}/envs/${environment.id}`;
        if (!fs_1.default.existsSync(`${dlcEnvironment.workingFolder}/commands`)) {
            await fs_1.default.promises.mkdir(`${dlcEnvironment.workingFolder}/commands`, { recursive: true });
        }
        environment.command = `cd ${environmentRoot} && ${environment.command} 1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr`;
        const dlcClient = new dlcClient_1.DlcClient(this.config.type, this.config.image, this.config.jobType, this.config.podCount, this.experimentId, environment.id, this.config.ecsSpec, this.config.region, this.config.workspaceId, this.config.nasDataSourceId, this.config.accessKeyId, this.config.accessKeySecret, environment.command, path_1.default.join((0, utils_1.getLogDir)(), `envs/${environment.id}`), this.config.ossDataSourceId);
        dlcEnvironment.id = await dlcClient.submit();
        this.log.debug('dlc: before getTrackingUrl');
        dlcEnvironment.trackingUrl = await dlcClient.getTrackingUrl();
        this.log.debug(`dlc trackingUrl: ${dlcEnvironment.trackingUrl}`);
        dlcEnvironment.dlcClient = dlcClient;
    }
    async stopEnvironment(environment) {
        const dlcEnvironment = environment;
        const dlcClient = dlcEnvironment.dlcClient;
        if (!dlcClient) {
            throw new Error('DLC client not initialized!');
        }
        dlcClient.stop();
    }
}
exports.DlcEnvironmentService = DlcEnvironmentService;
