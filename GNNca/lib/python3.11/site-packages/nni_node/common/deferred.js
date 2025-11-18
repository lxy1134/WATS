"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Deferred = void 0;
const util_1 = __importDefault(require("util"));
const log_1 = require("common/log");
const logger = (0, log_1.getLogger)('common.deferred');
class Deferred {
    resolveCallbacks = [];
    rejectCallbacks = [];
    isResolved = false;
    isRejected = false;
    resolvedValue;
    rejectedReason;
    get promise() {
        if (this.isResolved) {
            return Promise.resolve(this.resolvedValue);
        }
        if (this.isRejected) {
            return Promise.reject(this.rejectedReason);
        }
        return new Promise((resolutionFunc, rejectionFunc) => {
            this.resolveCallbacks.push(resolutionFunc);
            this.rejectCallbacks.push(rejectionFunc);
        });
    }
    get settled() {
        return this.isResolved || this.isRejected;
    }
    resolve = (value) => {
        if (!this.isResolved && !this.isRejected) {
            this.isResolved = true;
            this.resolvedValue = value;
            for (const callback of this.resolveCallbacks) {
                callback(value);
            }
        }
        else if (this.isResolved && this.resolvedValue === value) {
            logger.debug('Double resolve:', value);
        }
        else {
            const msg = this.errorMessage('trying to resolve with value: ' + util_1.default.inspect(value));
            logger.error(msg);
            throw new Error('Conflict Deferred result. ' + msg);
        }
    };
    reject = (reason) => {
        if (!this.isResolved && !this.isRejected) {
            this.isRejected = true;
            this.rejectedReason = reason;
            for (const callback of this.rejectCallbacks) {
                callback(reason);
            }
        }
        else if (this.isRejected) {
            logger.warning('Double reject:', this.rejectedReason, reason);
        }
        else {
            const msg = this.errorMessage('trying to reject with reason: ' + util_1.default.inspect(reason));
            logger.error(msg);
            throw new Error('Conflict Deferred result. ' + msg);
        }
    };
    errorMessage(curStat) {
        let prevStat = '';
        if (this.isResolved) {
            prevStat = 'Already resolved with value: ' + util_1.default.inspect(this.resolvedValue);
        }
        if (this.isRejected) {
            prevStat = 'Already rejected with reason: ' + util_1.default.inspect(this.rejectedReason);
        }
        return prevStat + ' ; ' + curStat;
    }
}
exports.Deferred = Deferred;
