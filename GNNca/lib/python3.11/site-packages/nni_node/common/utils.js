"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getNewLine = exports.killPid = exports.isAlive = exports.getTunerProc = exports.getVersion = exports.getLogLevel = exports.randomSelect = exports.randomInt = exports.uniqueString = exports.cleanupUnitTest = exports.prepareUnitTest = exports.delay = exports.mkDirPSync = exports.mkDirP = exports.isPortOpen = exports.getFreePort = exports.unixPathJoin = exports.getIPV4Address = exports.getDefaultDatabaseDir = exports.getJobCancelStatus = exports.getExperimentRootDir = exports.getLogDir = exports.getCheckpointDir = exports.getMsgDispatcherCommand = exports.generateParamFileName = exports.countFilesRecursively = exports.importModule = void 0;
const assert_1 = __importDefault(require("assert"));
const crypto_1 = require("crypto");
const child_process_promise_1 = __importDefault(require("child-process-promise"));
const child_process_1 = __importDefault(require("child_process"));
const child_process_2 = require("child_process");
const dgram_1 = __importDefault(require("dgram"));
const fs_1 = __importDefault(require("fs"));
const net_1 = __importDefault(require("net"));
const path_1 = __importDefault(require("path"));
const timersPromises = __importStar(require("timers/promises"));
const ts_deferred_1 = require("ts-deferred");
const datastore_1 = require("./datastore");
const globals_1 = __importDefault(require("./globals"));
const unittest_1 = require("./globals/unittest");
const ioc_shim_1 = require("./ioc_shim");
const manager_1 = require("./manager");
const trainingService_1 = require("./trainingService");
function getExperimentRootDir() {
    return globals_1.default.paths.experimentRoot;
}
exports.getExperimentRootDir = getExperimentRootDir;
function getLogDir() {
    return globals_1.default.paths.logDirectory;
}
exports.getLogDir = getLogDir;
function getLogLevel() {
    return globals_1.default.args.logLevel;
}
exports.getLogLevel = getLogLevel;
function getDefaultDatabaseDir() {
    return path_1.default.join(getExperimentRootDir(), 'db');
}
exports.getDefaultDatabaseDir = getDefaultDatabaseDir;
function getCheckpointDir() {
    return path_1.default.join(getExperimentRootDir(), 'checkpoint');
}
exports.getCheckpointDir = getCheckpointDir;
async function mkDirP(dirPath) {
    await fs_1.default.promises.mkdir(dirPath, { recursive: true });
}
exports.mkDirP = mkDirP;
function mkDirPSync(dirPath) {
    fs_1.default.mkdirSync(dirPath, { recursive: true });
}
exports.mkDirPSync = mkDirPSync;
const delay = timersPromises.setTimeout;
exports.delay = delay;
function charMap(index) {
    if (index < 26) {
        return index + 97;
    }
    else if (index < 52) {
        return index - 26 + 65;
    }
    else {
        return index - 52 + 48;
    }
}
function uniqueString(len) {
    if (len === 0) {
        return '';
    }
    const byteLength = Math.ceil((Math.log2(52) + Math.log2(62) * (len - 1)) / 8);
    let num = (0, crypto_1.randomBytes)(byteLength).reduce((a, b) => a * 256 + b, 0);
    const codes = [];
    codes.push(charMap(num % 52));
    num = Math.floor(num / 52);
    for (let i = 1; i < len; i++) {
        codes.push(charMap(num % 62));
        num = Math.floor(num / 62);
    }
    return String.fromCharCode(...codes);
}
exports.uniqueString = uniqueString;
function randomInt(max) {
    return Math.floor(Math.random() * max);
}
exports.randomInt = randomInt;
function randomSelect(a) {
    (0, assert_1.default)(a !== undefined);
    return a[Math.floor(Math.random() * a.length)];
}
exports.randomSelect = randomSelect;
function getMsgDispatcherCommand(expParams) {
    const clonedParams = Object.assign({}, expParams);
    delete clonedParams.searchSpace;
    return [globals_1.default.args.pythonInterpreter, '-m', 'nni', '--exp_params', Buffer.from(JSON.stringify(clonedParams)).toString('base64')];
}
exports.getMsgDispatcherCommand = getMsgDispatcherCommand;
function generateParamFileName(hyperParameters) {
    (0, assert_1.default)(hyperParameters !== undefined);
    (0, assert_1.default)(hyperParameters.index >= 0);
    let paramFileName;
    if (hyperParameters.index == 0) {
        paramFileName = 'parameter.cfg';
    }
    else {
        paramFileName = `parameter_${hyperParameters.index}.cfg`;
    }
    return paramFileName;
}
exports.generateParamFileName = generateParamFileName;
function prepareUnitTest() {
    ioc_shim_1.IocShim.snapshot(datastore_1.Database);
    ioc_shim_1.IocShim.snapshot(datastore_1.DataStore);
    ioc_shim_1.IocShim.snapshot(trainingService_1.TrainingService);
    ioc_shim_1.IocShim.snapshot(manager_1.Manager);
    (0, unittest_1.resetGlobals)();
    const sqliteFile = path_1.default.join(getDefaultDatabaseDir(), 'nni.sqlite');
    try {
        fs_1.default.unlinkSync(sqliteFile);
    }
    catch (err) {
    }
}
exports.prepareUnitTest = prepareUnitTest;
function cleanupUnitTest() {
    ioc_shim_1.IocShim.restore(manager_1.Manager);
    ioc_shim_1.IocShim.restore(trainingService_1.TrainingService);
    ioc_shim_1.IocShim.restore(datastore_1.DataStore);
    ioc_shim_1.IocShim.restore(datastore_1.Database);
}
exports.cleanupUnitTest = cleanupUnitTest;
let cachedIpv4Address = null;
async function getIPV4Address() {
    if (cachedIpv4Address !== null) {
        return cachedIpv4Address;
    }
    const socket = dgram_1.default.createSocket('udp4');
    socket.connect(1, '192.0.2.0');
    for (let i = 0; i < 10; i++) {
        await timersPromises.setTimeout(1);
        try {
            cachedIpv4Address = socket.address().address;
            socket.close();
            return cachedIpv4Address;
        }
        catch (error) {
        }
    }
    cachedIpv4Address = socket.address().address;
    socket.close();
    return cachedIpv4Address;
}
exports.getIPV4Address = getIPV4Address;
function getJobCancelStatus(isEarlyStopped) {
    return isEarlyStopped ? 'EARLY_STOPPED' : 'USER_CANCELED';
}
exports.getJobCancelStatus = getJobCancelStatus;
function countFilesRecursively(directory) {
    if (!fs_1.default.existsSync(directory)) {
        throw Error(`Direcotory ${directory} doesn't exist`);
    }
    const deferred = new ts_deferred_1.Deferred();
    let timeoutId;
    const delayTimeout = new Promise((_resolve, reject) => {
        timeoutId = setTimeout(() => {
            reject(new Error(`Timeout: path ${directory} has too many files`));
        }, 5000);
    });
    let fileCount = -1;
    let cmd;
    if (process.platform === "win32") {
        cmd = `powershell "Get-ChildItem -Path ${directory} -Recurse -File | Measure-Object | %{$_.Count}"`;
    }
    else {
        cmd = `find ${directory} -type f | wc -l`;
    }
    child_process_promise_1.default.exec(cmd).then((result) => {
        if (result.stdout && parseInt(result.stdout)) {
            fileCount = parseInt(result.stdout);
        }
        deferred.resolve(fileCount);
    });
    return Promise.race([deferred.promise, delayTimeout]).finally(() => {
        clearTimeout(timeoutId);
    });
}
exports.countFilesRecursively = countFilesRecursively;
async function getVersion() {
    const deferred = new ts_deferred_1.Deferred();
    Promise.resolve().then(() => __importStar(require(path_1.default.join(__dirname, '..', 'package.json')))).then((pkg) => {
        deferred.resolve(pkg.version);
    }).catch(() => {
        deferred.resolve('999.0.0-developing');
    });
    return deferred.promise;
}
exports.getVersion = getVersion;
function getTunerProc(command, stdio, newCwd, newEnv, newShell = true, isDetached = false) {
    if (process.platform === "win32") {
        newShell = false;
        isDetached = true;
    }
    const tunerProc = (0, child_process_2.spawn)(command[0], command.slice(1), {
        stdio,
        cwd: newCwd,
        env: newEnv,
        shell: newShell,
        detached: isDetached
    });
    return tunerProc;
}
exports.getTunerProc = getTunerProc;
async function isAlive(pid) {
    const deferred = new ts_deferred_1.Deferred();
    let alive = false;
    if (process.platform === 'win32') {
        try {
            const str = child_process_1.default.execSync(`powershell.exe Get-Process -Id ${pid} -ErrorAction SilentlyContinue`).toString();
            if (str) {
                alive = true;
            }
        }
        catch (error) {
        }
    }
    else {
        try {
            await child_process_promise_1.default.exec(`kill -0 ${pid}`);
            alive = true;
        }
        catch (error) {
        }
    }
    deferred.resolve(alive);
    return deferred.promise;
}
exports.isAlive = isAlive;
async function killPid(pid) {
    const deferred = new ts_deferred_1.Deferred();
    try {
        if (process.platform === "win32") {
            await child_process_promise_1.default.exec(`cmd.exe /c taskkill /PID ${pid} /F`);
        }
        else {
            await child_process_promise_1.default.exec(`kill -9 ${pid}`);
        }
    }
    catch (error) {
    }
    deferred.resolve();
    return deferred.promise;
}
exports.killPid = killPid;
function getNewLine() {
    if (process.platform === "win32") {
        return "\r\n";
    }
    else {
        return "\n";
    }
}
exports.getNewLine = getNewLine;
function unixPathJoin(...paths) {
    const dir = paths.filter((path) => path !== '').join('/');
    if (dir === '')
        return '.';
    return dir;
}
exports.unixPathJoin = unixPathJoin;
async function isPortOpen(host, port) {
    return new Promise((resolve, reject) => {
        try {
            const stream = net_1.default.createConnection(port, host);
            const id = setTimeout(() => {
                stream.destroy();
                resolve(false);
            }, 1000);
            stream.on('connect', () => {
                clearTimeout(id);
                stream.destroy();
                resolve(true);
            });
            stream.on('error', () => {
                clearTimeout(id);
                stream.destroy();
                resolve(false);
            });
        }
        catch (error) {
            reject(error);
        }
    });
}
exports.isPortOpen = isPortOpen;
async function getFreePort(host, start, end) {
    if (start > end) {
        throw new Error(`no more free port`);
    }
    if (await isPortOpen(host, start)) {
        return await getFreePort(host, start + 1, end);
    }
    else {
        return start;
    }
}
exports.getFreePort = getFreePort;
function importModule(modulePath) {
    module.paths.unshift(path_1.default.dirname(modulePath));
    return require(path_1.default.basename(modulePath));
}
exports.importModule = importModule;
