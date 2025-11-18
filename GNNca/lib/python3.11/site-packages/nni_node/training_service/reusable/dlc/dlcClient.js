"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DlcClient = void 0;
const ts_deferred_1 = require("ts-deferred");
const python_shell_1 = require("python-shell");
const log_1 = require("common/log");
class DlcClient {
    log;
    type;
    image;
    jobType;
    podCount;
    ecsSpec;
    region;
    workspaceId;
    nasDataSourceId;
    ossDataSourceId;
    accessKeyId;
    accessKeySecret;
    experimentId;
    environmentId;
    userCommand;
    logDir;
    pythonShellClient;
    status;
    constructor(type, image, jobType, podCount, experimentId, environmentId, ecsSpec, region, workspaceId, nasDataSourceId, accessKeyId, accessKeySecret, userCommand, logDir, ossDataSourceId) {
        this.log = (0, log_1.getLogger)('DlcClient');
        this.type = type;
        this.image = image;
        this.jobType = jobType;
        this.podCount = podCount;
        this.ecsSpec = ecsSpec;
        this.image = image;
        this.region = region;
        this.workspaceId = workspaceId;
        this.nasDataSourceId = nasDataSourceId;
        if (ossDataSourceId !== undefined) {
            this.ossDataSourceId = ossDataSourceId;
        }
        else {
            this.ossDataSourceId = '';
        }
        this.accessKeyId = accessKeyId;
        this.accessKeySecret = accessKeySecret;
        this.experimentId = experimentId;
        this.environmentId = environmentId;
        this.userCommand = userCommand;
        this.logDir = logDir;
        this.status = '';
    }
    submit() {
        const deferred = new ts_deferred_1.Deferred();
        this.pythonShellClient = new python_shell_1.PythonShell('dlcUtil.py', {
            scriptPath: './config/dlc',
            pythonPath: 'python3',
            pythonOptions: ['-u'],
            args: [
                '--type', this.type,
                '--image', this.image,
                '--job_type', this.jobType,
                '--pod_count', String(this.podCount),
                '--ecs_spec', this.ecsSpec,
                '--region', this.region,
                '--workspace_id', this.workspaceId,
                '--nas_data_source_id', this.nasDataSourceId,
                '--oss_data_source_id', this.ossDataSourceId,
                '--access_key_id', this.accessKeyId,
                '--access_key_secret', this.accessKeySecret,
                '--experiment_name', `nni_exp_${this.experimentId}_env_${this.environmentId}`,
                '--user_command', this.userCommand,
                '--log_dir', this.logDir,
            ]
        });
        this.log.debug(this.pythonShellClient.command);
        this.onMessage();
        this.log.debug(`on message`);
        this.monitorError(this.pythonShellClient, deferred);
        this.log.debug(`monitor submit`);
        const log = this.log;
        this.pythonShellClient.on('message', (message) => {
            const jobid = this.parseContent('job_id', message);
            if (jobid !== '') {
                log.debug(`reslove job_id ${jobid}`);
                deferred.resolve(jobid);
            }
        });
        return deferred.promise;
    }
    onMessage() {
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        const log = this.log;
        this.pythonShellClient.on('message', (message) => {
            const status = this.parseContent('status', message);
            if (status.length > 0) {
                log.debug(`on message status: ${status}`);
                this.status = status;
                return;
            }
        });
    }
    stop() {
        if (this.pythonShellClient === undefined) {
            this.log.debug(`python shell client not initialized!`);
            throw Error('python shell client not initialized!');
        }
        this.log.debug(`send stop`);
        this.pythonShellClient.send('stop');
    }
    getTrackingUrl() {
        const deferred = new ts_deferred_1.Deferred();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.log.debug(`send tracking_url`);
        this.pythonShellClient.send('tracking_url');
        const log = this.log;
        this.pythonShellClient.on('message', (status) => {
            const trackingUrl = this.parseContent('tracking_url', status);
            if (trackingUrl !== '') {
                log.debug(`trackingUrl:${trackingUrl}`);
                deferred.resolve(trackingUrl);
            }
        });
        return deferred.promise;
    }
    updateStatus(oldStatus) {
        const deferred = new ts_deferred_1.Deferred();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('update_status');
        if (this.status === '') {
            this.status = oldStatus;
        }
        this.log.debug(`update_status:${this.status}`);
        deferred.resolve(this.status);
        return deferred.promise;
    }
    monitorError(pythonShellClient, deferred) {
        const log = this.log;
        pythonShellClient.on('error', function (error) {
            log.info(`error:${error}`);
            deferred.reject(error);
        });
    }
    parseContent(head, command) {
        const items = command.split(':');
        if (items[0] === head) {
            return command.slice(head.length + 1);
        }
        return '';
    }
}
exports.DlcClient = DlcClient;
