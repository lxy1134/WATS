"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LocalEnvironmentService = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const tree_kill_1 = __importDefault(require("tree-kill"));
const ioc_shim_1 = require("common/ioc_shim");
const log_1 = require("common/log");
const shellUtils_1 = require("common/shellUtils");
const environment_1 = require("../environment");
const utils_1 = require("common/utils");
const util_1 = require("training_service/common/util");
const sharedStorage_1 = require("../sharedStorage");
class LocalEnvironmentService extends environment_1.EnvironmentService {
    log = (0, log_1.getLogger)('LocalEnvironmentService');
    experimentRootDir;
    experimentId;
    constructor(_config, info) {
        super();
        this.experimentId = info.experimentId;
        this.experimentRootDir = info.logDir;
    }
    get environmentMaintenceLoopInterval() {
        return 100;
    }
    get hasStorageService() {
        return false;
    }
    get getName() {
        return 'local';
    }
    async refreshEnvironmentsStatus(environments) {
        environments.forEach(async (environment) => {
            const jobpidPath = `${path_1.default.join(environment.runnerWorkingFolder, 'pid')}`;
            const runnerReturnCodeFilePath = `${path_1.default.join(environment.runnerWorkingFolder, 'code')}`;
            try {
                const pidExist = await fs_1.default.existsSync(jobpidPath);
                if (!pidExist) {
                    return;
                }
                const pid = await fs_1.default.promises.readFile(jobpidPath, 'utf8');
                const alive = await (0, utils_1.isAlive)(pid);
                environment.status = 'RUNNING';
                if (!alive) {
                    if (fs_1.default.existsSync(runnerReturnCodeFilePath)) {
                        const runnerReturnCode = await fs_1.default.promises.readFile(runnerReturnCodeFilePath, 'utf8');
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
                        }
                    }
                }
            }
            catch (error) {
                this.log.error(`Update job status exception, error is ${error.message}`);
            }
        });
    }
    getScript(environment) {
        const script = [];
        if (process.platform === 'win32') {
            script.push(`$env:PATH=${(0, shellUtils_1.powershellString)(process.env['path'])}`);
            script.push(`cd $env:${this.experimentRootDir}`);
            script.push(`New-Item -ItemType "directory" -Path ${path_1.default.join(this.experimentRootDir, 'envs', environment.id)} -Force`);
            script.push(`cd envs\\${environment.id}`);
            environment.command = `python -m nni.tools.trial_tool.trial_runner`;
            script.push(`cmd.exe /c ${environment.command} --job_pid_file ${path_1.default.join(environment.runnerWorkingFolder, 'pid')} 2>&1 | Out-File "${path_1.default.join(environment.runnerWorkingFolder, 'trial_runner.log')}" -encoding utf8`, `$NOW_DATE = [int64](([datetime]::UtcNow)-(get-date "1/1/1970")).TotalSeconds`, `$NOW_DATE = "$NOW_DATE" + (Get-Date -Format fff).ToString()`, `Write $LASTEXITCODE " " $NOW_DATE  | Out-File "${path_1.default.join(environment.runnerWorkingFolder, 'code')}" -NoNewline -encoding utf8`);
        }
        else {
            script.push(`cd ${this.experimentRootDir}`);
            script.push(`eval ${environment.command} --job_pid_file ${environment.runnerWorkingFolder}/pid 1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr`);
            if (process.platform === 'darwin') {
                script.push(`echo $? \`date +%s999\` >'${environment.runnerWorkingFolder}/code'`);
            }
            else {
                script.push(`echo $? \`date +%s%3N\` >'${environment.runnerWorkingFolder}/code'`);
            }
        }
        return script;
    }
    async startEnvironment(environment) {
        const sharedStorageService = ioc_shim_1.IocShim.get(sharedStorage_1.SharedStorageService);
        if (environment.useSharedStorage && sharedStorageService.canLocalMounted) {
            this.experimentRootDir = sharedStorageService.localWorkingRoot;
        }
        const localEnvCodeFolder = path_1.default.join(this.experimentRootDir, "envs");
        if (environment.useSharedStorage && !sharedStorageService.canLocalMounted) {
            await sharedStorageService.storageService.copyDirectoryBack("envs", localEnvCodeFolder);
        }
        else if (!environment.useSharedStorage) {
            const localTempFolder = path_1.default.join(this.experimentRootDir, "environment-temp", "envs");
            await (0, util_1.execCopydir)(localTempFolder, localEnvCodeFolder);
        }
        environment.runnerWorkingFolder = path_1.default.join(localEnvCodeFolder, environment.id);
        await (0, util_1.execMkdir)(environment.runnerWorkingFolder);
        environment.command = this.getScript(environment).join((0, utils_1.getNewLine)());
        const scriptName = (0, util_1.getScriptName)('run');
        await (0, shellUtils_1.createScriptFile)(path_1.default.join(localEnvCodeFolder, scriptName), environment.command);
        (0, util_1.runScript)(path_1.default.join(localEnvCodeFolder, scriptName));
        environment.trackingUrl = `${environment.runnerWorkingFolder}`;
    }
    async stopEnvironment(environment) {
        if (environment.isAlive === false) {
            return Promise.resolve();
        }
        const jobpidPath = `${path_1.default.join(environment.runnerWorkingFolder, 'pid')}`;
        const pid = await fs_1.default.promises.readFile(jobpidPath, 'utf8');
        (0, tree_kill_1.default)(Number(pid), 'SIGKILL');
    }
}
exports.LocalEnvironmentService = LocalEnvironmentService;
