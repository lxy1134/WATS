"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runPython = exports.runPythonModule = exports.runPythonScript = void 0;
const child_process_1 = require("child_process");
const globals_1 = __importDefault(require("./globals"));
const log_1 = require("./log");
const logger = (0, log_1.getLogger)('pythonScript');
function runPythonScript(script, logTag) {
    return runPython(['-c', script], logTag);
}
exports.runPythonScript = runPythonScript;
function runPythonModule(moduleName, args) {
    const argsArr = args ?? [];
    return runPython(['-m', moduleName, ...argsArr], moduleName);
}
exports.runPythonModule = runPythonModule;
async function runPython(args, logTag) {
    const proc = (0, child_process_1.spawn)(globals_1.default.args.pythonInterpreter, args);
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (data) => { stdout += data; });
    proc.stderr.on('data', (data) => { stderr += data; });
    const procPromise = new Promise((resolve, reject) => {
        proc.on('error', (err) => { reject(err); });
        proc.on('exit', () => { resolve(); });
    });
    await procPromise;
    if (stderr) {
        if (logTag) {
            logger.warning(`Python command [${logTag}] has stderr:`, stderr);
        }
        else {
            logger.warning('Python command has stderr.');
            logger.warning('  args:', args);
            logger.warning('  stderr:', stderr);
        }
    }
    return stdout;
}
exports.runPython = runPython;
