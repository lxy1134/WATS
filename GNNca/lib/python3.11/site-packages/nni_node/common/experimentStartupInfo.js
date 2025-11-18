"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.isReadonly = exports.getPlatform = exports.isNewExperiment = exports.getBasePort = exports.getExperimentId = exports.getExperimentStartupInfo = exports.ExperimentStartupInfo = void 0;
const globals_1 = __importDefault(require("common/globals"));
class ExperimentStartupInfo {
    experimentId = globals_1.default.args.experimentId;
    newExperiment = (globals_1.default.args.action === 'create');
    basePort = globals_1.default.args.port;
    logDir = globals_1.default.paths.experimentRoot;
    logLevel = globals_1.default.args.logLevel;
    readonly = (globals_1.default.args.action === 'view');
    platform = globals_1.default.args.mode;
    urlprefix = globals_1.default.args.urlPrefix;
    static getInstance() {
        return new ExperimentStartupInfo();
    }
}
exports.ExperimentStartupInfo = ExperimentStartupInfo;
function getExperimentStartupInfo() {
    return new ExperimentStartupInfo();
}
exports.getExperimentStartupInfo = getExperimentStartupInfo;
function getExperimentId() {
    return globals_1.default.args.experimentId;
}
exports.getExperimentId = getExperimentId;
function getBasePort() {
    return globals_1.default.args.port;
}
exports.getBasePort = getBasePort;
function isNewExperiment() {
    return globals_1.default.args.action === 'create';
}
exports.isNewExperiment = isNewExperiment;
function getPlatform() {
    return globals_1.default.args.mode;
}
exports.getPlatform = getPlatform;
function isReadonly() {
    return globals_1.default.args.action === 'view';
}
exports.isReadonly = isReadonly;
