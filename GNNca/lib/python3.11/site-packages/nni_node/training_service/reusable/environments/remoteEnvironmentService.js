"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RemoteEnvironmentService = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const ioc_shim_1 = require("common/ioc_shim");
const log_1 = require("common/log");
const environment_1 = require("../environment");
const utils_1 = require("common/utils");
const util_1 = require("training_service/common/util");
const remoteMachineData_1 = require("training_service/remote_machine/remoteMachineData");
const sharedStorage_1 = require("../sharedStorage");
const shellUtils_1 = require("common/shellUtils");
class RemoteEnvironmentService extends environment_1.EnvironmentService {
    initExecutorId = "initConnection";
    machineExecutorManagerMap;
    environmentExecutorManagerMap;
    remoteMachineMetaOccupiedMap;
    log;
    sshConnectionPromises;
    experimentRootDir;
    remoteExperimentRootDir = "";
    experimentId;
    config;
    constructor(config, info) {
        super();
        this.experimentId = info.experimentId;
        this.environmentExecutorManagerMap = new Map();
        this.machineExecutorManagerMap = new Map();
        this.remoteMachineMetaOccupiedMap = new Map();
        this.experimentRootDir = info.logDir;
        this.log = (0, log_1.getLogger)('RemoteEnvironmentService');
        this.config = config;
        if (!fs_1.default.lstatSync(this.config.trialCodeDirectory).isDirectory()) {
            throw new Error(`codeDir ${this.config.trialCodeDirectory} is not a directory`);
        }
        this.sshConnectionPromises = Promise.all(this.config.machineList.map(machine => this.initRemoteMachineOnConnected(machine)));
    }
    async init() {
        await this.sshConnectionPromises;
        this.log.info('ssh connection initialized!');
        Array.from(this.machineExecutorManagerMap.keys()).forEach(rmMeta => {
            this.remoteMachineMetaOccupiedMap.set(rmMeta, false);
        });
    }
    get prefetchedEnvironmentCount() {
        return this.machineExecutorManagerMap.size;
    }
    get environmentMaintenceLoopInterval() {
        return 5000;
    }
    get hasMoreEnvironments() {
        return false;
    }
    get hasStorageService() {
        return false;
    }
    get getName() {
        return 'remote';
    }
    scheduleMachine() {
        for (const [rmMeta, occupied] of this.remoteMachineMetaOccupiedMap) {
            if (!occupied) {
                this.remoteMachineMetaOccupiedMap.set(rmMeta, true);
                return rmMeta;
            }
        }
        return undefined;
    }
    async initRemoteMachineOnConnected(rmMeta) {
        const executorManager = new remoteMachineData_1.ExecutorManager(rmMeta);
        this.log.info(`connecting to ${rmMeta.user}@${rmMeta.host}:${rmMeta.port}`);
        const executor = await executorManager.getExecutor(this.initExecutorId);
        this.log.debug(`reached ${executor.name}`);
        this.machineExecutorManagerMap.set(rmMeta, executorManager);
        this.log.debug(`initializing ${executor.name}`);
        const nniRootDir = executor.joinPath(executor.getTempPath(), 'nni-experiments');
        await executor.createFolder(executor.getRemoteExperimentRootDir(this.experimentId));
        const remoteGpuScriptCollectorDir = executor.getRemoteScriptsPath(this.experimentId);
        await executor.createFolder(remoteGpuScriptCollectorDir, true);
        await executor.allowPermission(true, nniRootDir);
    }
    async refreshEnvironmentsStatus(environments) {
        const tasks = environments.map(environment => this.refreshEnvironment(environment));
        await Promise.all(tasks);
    }
    async refreshEnvironment(environment) {
        const executor = await this.getExecutor(environment.id);
        const jobpidPath = `${environment.runnerWorkingFolder}/pid`;
        const runnerReturnCodeFilePath = `${environment.runnerWorkingFolder}/code`;
        try {
            const pidExist = await executor.fileExist(jobpidPath);
            if (!pidExist) {
                return;
            }
            const isAlive = await executor.isProcessAlive(jobpidPath);
            environment.status = 'RUNNING';
            if (!isAlive) {
                const remoteEnvironment = environment;
                if (remoteEnvironment.rmMachineMeta === undefined) {
                    throw new Error(`${remoteEnvironment.id} machine meta not initialized!`);
                }
                this.log.info(`pid in ${remoteEnvironment.rmMachineMeta.host}:${jobpidPath} is not alive!`);
                if (fs_1.default.existsSync(runnerReturnCodeFilePath)) {
                    const runnerReturnCode = await executor.getRemoteFileContent(runnerReturnCodeFilePath);
                    const match = runnerReturnCode.trim()
                        .match(/^-?(\d+)\s+(\d+)$/);
                    if (match !== null) {
                        const { 1: code } = match;
                        if (parseInt(code, 10) === 0) {
                            environment.setStatus('SUCCEEDED');
                        }
                        else {
                            environment.setStatus('FAILED');
                        }
                        await this.releaseEnvironmentResource(environment);
                    }
                }
            }
        }
        catch (error) {
            this.log.error(`Update job status exception, error is ${error.message}`);
        }
    }
    async releaseEnvironmentResource(environment) {
        if (environment.useSharedStorage) {
            const executor = await this.getExecutor(environment.id);
            const remoteUmountCommand = ioc_shim_1.IocShim.get(sharedStorage_1.SharedStorageService).remoteUmountCommand;
            const result = await executor.executeScript(remoteUmountCommand, false, false);
            if (result.exitCode !== 0) {
                this.log.error(`Umount shared storage on remote machine failed.\n ERROR: ${result.stderr}`);
            }
        }
        const executorManager = this.environmentExecutorManagerMap.get(environment.id);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for environment ${environment.id}`);
        }
        executorManager.releaseExecutor(environment.id);
        const remoteEnvironment = environment;
        if (remoteEnvironment.rmMachineMeta === undefined) {
            throw new Error(`${remoteEnvironment.id} rmMachineMeta not initialized!`);
        }
        this.remoteMachineMetaOccupiedMap.set(remoteEnvironment.rmMachineMeta, false);
    }
    async getScript(environment) {
        const executor = await this.getExecutor(environment.id);
        const isDebug = (0, utils_1.getLogLevel)() == "debug";
        let script = environment.command;
        environment.runnerWorkingFolder = executor.joinPath(this.remoteExperimentRootDir, 'envs', environment.id);
        let codeScript = `echo $? \`date +%s%3N\` >${environment.runnerWorkingFolder}/code`;
        if (executor.isWindows) {
            const prepare = `mkdir envs\\${environment.id} 2>NUL & cd envs\\${environment.id}`;
            const startrun = `powershell ..\\install_nni.ps1 && python -m nni.tools.trial_tool.trial_runner`;
            const developingScript = "IF EXIST nni_trial_tool (ECHO \"nni_trial_tool exists already\") ELSE (mkdir nni_trial_tool && tar -xof ../nni_trial_tool.tar.gz -C ./nni_trial_tool) && pip3 install websockets";
            script = isDebug ? `${prepare} && ${developingScript} && ${startrun}` : `${prepare} && ${startrun}`;
            codeScript = `powershell -command "Write $? " " (((New-TimeSpan -Start (Get-Date "01/01/1970") -End (Get-Date).ToUniversalTime()).TotalMilliseconds).ToString("0")) | Out-file ${path_1.default.join(environment.runnerWorkingFolder, 'code')} -Append -NoNewline -encoding utf8"`;
        }
        script = `cd ${this.remoteExperimentRootDir} && \
            ${script} --job_pid_file ${environment.runnerWorkingFolder}/pid \
            1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr \
            && ${codeScript}`;
        return script;
    }
    async startEnvironment(environment) {
        const remoteEnvironment = environment;
        remoteEnvironment.status = 'WAITING';
        await this.prepareEnvironment(remoteEnvironment);
        await this.launchRunner(environment);
    }
    async prepareEnvironment(environment) {
        const rmMachineMeta = this.scheduleMachine();
        if (rmMachineMeta === undefined) {
            this.log.warning(`No available machine!`);
            return Promise.resolve(false);
        }
        else {
            environment.rmMachineMeta = rmMachineMeta;
            const executorManager = this.machineExecutorManagerMap.get(environment.rmMachineMeta);
            if (executorManager === undefined) {
                throw new Error(`executorManager not initialized`);
            }
            this.environmentExecutorManagerMap.set(environment.id, executorManager);
            const executor = await this.getExecutor(environment.id);
            if (environment.useSharedStorage) {
                this.remoteExperimentRootDir = ioc_shim_1.IocShim.get(sharedStorage_1.SharedStorageService).remoteWorkingRoot;
                if (!this.remoteExperimentRootDir.startsWith('/')) {
                    this.remoteExperimentRootDir = executor.joinPath((await executor.getCurrentPath()).trim(), this.remoteExperimentRootDir);
                }
                const remoteMountCommand = ioc_shim_1.IocShim.get(sharedStorage_1.SharedStorageService).remoteMountCommand.replace(/echo -e /g, `echo `).replace(/echo /g, `echo -e `).replace(/\\\$/g, `\\\\\\$`);
                const result = await executor.executeScript(remoteMountCommand, false, false);
                if (result.exitCode !== 0) {
                    throw new Error(`Mount shared storage on remote machine failed.\n ERROR: ${result.stderr}`);
                }
            }
            else {
                this.remoteExperimentRootDir = executor.getRemoteExperimentRootDir(this.experimentId);
            }
            environment.command = await this.getScript(environment);
            environment.useActiveGpu = rmMachineMeta.useActiveGpu;
            return Promise.resolve(true);
        }
    }
    async launchRunner(environment) {
        const executor = await this.getExecutor(environment.id);
        const environmentLocalTempFolder = path_1.default.join(this.experimentRootDir, "environment-temp");
        await executor.createFolder(environment.runnerWorkingFolder);
        await (0, util_1.execMkdir)(environmentLocalTempFolder);
        await (0, shellUtils_1.createScriptFile)(path_1.default.join(environmentLocalTempFolder, executor.getScriptName("run")), environment.command);
        await executor.copyDirectoryToRemote(environmentLocalTempFolder, this.remoteExperimentRootDir);
        executor.executeScript(executor.joinPath(this.remoteExperimentRootDir, executor.getScriptName("run")), true, true);
        if (environment.rmMachineMeta === undefined) {
            throw new Error(`${environment.id} rmMachineMeta not initialized!`);
        }
        environment.trackingUrl = `file://${environment.rmMachineMeta.host}:${environment.runnerWorkingFolder}`;
    }
    async getExecutor(environmentId) {
        const executorManager = this.environmentExecutorManagerMap.get(environmentId);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for environment ${environmentId}`);
        }
        return await executorManager.getExecutor(environmentId);
    }
    async stopEnvironment(environment) {
        if (environment.isAlive === false) {
            return;
        }
        const executor = await this.getExecutor(environment.id);
        if (environment.status === 'UNKNOWN') {
            environment.status = 'USER_CANCELED';
            await this.releaseEnvironmentResource(environment);
            return;
        }
        const jobpidPath = `${environment.runnerWorkingFolder}/pid`;
        try {
            await executor.killChildProcesses(jobpidPath);
            await this.releaseEnvironmentResource(environment);
        }
        catch (error) {
            this.log.error(`stopEnvironment: ${error}`);
        }
    }
}
exports.RemoteEnvironmentService = RemoteEnvironmentService;
