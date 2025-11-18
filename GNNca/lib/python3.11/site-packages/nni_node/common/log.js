"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Logger = exports.getRobustLogger = exports.getLogger = void 0;
const util_1 = __importDefault(require("util"));
const globals_1 = __importDefault(require("common/globals"));
const levelNameToValue = { trace: 0, debug: 10, info: 20, warning: 30, error: 40, critical: 50 };
const loggers = {};
function getLogger(name) {
    if (loggers[name] === undefined) {
        loggers[name] = new Logger(name);
    }
    return loggers[name];
}
exports.getLogger = getLogger;
function getRobustLogger(name) {
    if (loggers[name] === undefined || !loggers[name].robust) {
        loggers[name] = new RobustLogger(name);
    }
    return loggers[name];
}
exports.getRobustLogger = getRobustLogger;
class Logger {
    name;
    constructor(name) {
        this.name = name;
    }
    trace(...args) {
        this.log(levelNameToValue.trace, 'TRACE', args);
    }
    debug(...args) {
        this.log(levelNameToValue.debug, 'DEBUG', args);
    }
    info(...args) {
        this.log(levelNameToValue.info, 'INFO', args);
    }
    warning(...args) {
        this.log(levelNameToValue.warning, 'WARNING', args);
    }
    error(...args) {
        this.log(levelNameToValue.error, 'ERROR', args);
    }
    critical(...args) {
        this.log(levelNameToValue.critical, 'CRITICAL', args);
    }
    log(levelValue, levelName, args) {
        if (levelValue >= levelNameToValue[globals_1.default.args.logLevel]) {
            const msg = `[${timestamp()}] ${levelName} (${this.name}) ${formatArgs(args)}`;
            globals_1.default.logStream.writeLine(msg);
        }
    }
}
exports.Logger = Logger;
class RobustLogger extends Logger {
    robust = true;
    errorOccurred = false;
    log(levelValue, levelName, args) {
        if (this.errorOccurred) {
            this.logAfterError(levelName, args);
            return;
        }
        try {
            if (levelValue >= levelNameToValue[globals_1.default.args.logLevel]) {
                const msg = `[${timestamp()}] ${levelName} (${this.name}) ${formatArgs(args)}`;
                globals_1.default.logStream.writeLineSync(msg);
            }
        }
        catch (error) {
            this.errorOccurred = true;
            console.error('[ERROR] Logger has stopped working:', error);
            this.logAfterError(levelName, args);
        }
    }
    logAfterError(levelName, args) {
        try {
            args = args.map(arg => util_1.default.inspect(arg));
        }
        catch { }
        console.error(`[${levelName}] (${this.name})`, ...args);
    }
}
function timestamp() {
    const now = new Date();
    const date = now.getFullYear() + '-' + zeroPad(now.getMonth() + 1) + '-' + zeroPad(now.getDate());
    const time = zeroPad(now.getHours()) + ':' + zeroPad(now.getMinutes()) + ':' + zeroPad(now.getSeconds());
    return date + ' ' + time;
}
function zeroPad(num) {
    return num.toString().padStart(2, '0');
}
function formatArgs(args) {
    return args.map(arg => (typeof arg === 'string' ? arg : util_1.default.inspect(arg))).join(' ');
}
