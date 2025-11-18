"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelper = exports.WsChannel = void 0;
const node_events_1 = require("node:events");
const node_util_1 = __importDefault(require("node:util"));
const deferred_1 = require("common/deferred");
const log_1 = require("common/log");
const connection_1 = require("./connection");
class WsChannel {
    closing = false;
    connection = null;
    epoch = -1;
    heartbeatInterval = null;
    log;
    queue = [];
    terminateTimer = null;
    emitter = new node_events_1.EventEmitter();
    name;
    constructor(name) {
        this.log = (0, log_1.getLogger)(`WsChannel.${name}`);
        this.name = name;
    }
    async setConnection(ws, waitOpen) {
        if (this.terminateTimer) {
            clearTimeout(this.terminateTimer);
            this.terminateTimer = null;
        }
        this.connection?.terminate('new epoch start');
        this.newEpoch();
        this.log.debug(`Epoch ${this.epoch} start`);
        this.connection = this.configConnection(ws);
        if (waitOpen) {
            await (0, node_events_1.once)(ws, 'open');
        }
        while (this.connection && this.queue.length > 0) {
            const item = this.queue.shift();
            try {
                await this.connection.sendAsync(item.command);
                item.deferred?.resolve();
            }
            catch (error) {
                this.log.error('Failed to send command on recovered channel:', error);
                this.log.error('Dropped command:', item.command);
                item.deferred?.reject(error);
            }
        }
    }
    enableHeartbeat(interval) {
        this.heartbeatInterval = interval ?? defaultHeartbeatInterval;
        this.connection?.setHeartbeatInterval(this.heartbeatInterval);
    }
    close(reason) {
        this.log.debug('Close channel:', reason);
        this.connection?.close(reason);
        if (this.setClosing()) {
            this.emitter.emit('__close', reason);
        }
    }
    terminate(reason) {
        this.log.info('Terminate channel:', reason);
        this.connection?.terminate(reason);
        if (this.setClosing()) {
            this.emitter.emit('__error', new Error(`WsChannel terminated: ${reason}`));
        }
    }
    send(command) {
        if (this.closing) {
            this.log.error('Channel closed. Ignored command', command);
            return;
        }
        if (!this.connection) {
            this.log.warning('Connection lost. Enqueue command', command);
            this.queue.push({ command });
            return;
        }
        this.connection.send(command);
    }
    sendAsync(command) {
        if (this.closing) {
            this.log.error('(async) Channel closed. Refused command', command);
            return Promise.reject(new Error('WsChannel has been closed'));
        }
        if (!this.connection) {
            this.log.warning('(async) Connection lost. Enqueue command', command);
            const deferred = new deferred_1.Deferred();
            this.queue.push({ command, deferred });
            return deferred.promise;
        }
        return this.connection.sendAsync(command);
    }
    onReceive(callback) {
        this.emitter.on('__receive', callback);
    }
    onCommand(commandType, callback) {
        this.emitter.on(commandType, callback);
    }
    onClose(callback) {
        this.emitter.on('__close', callback);
    }
    onError(callback) {
        this.emitter.on('__error', callback);
    }
    onLost(callback) {
        this.emitter.on('__lost', callback);
    }
    getBufferedAmount() {
        return this.connection?.ws.bufferedAmount ?? 0;
    }
    newEpoch() {
        this.connection = null;
        this.epoch += 1;
    }
    configConnection(ws) {
        const connName = this.epoch ? `${this.name}.${this.epoch}` : this.name;
        const conn = new connection_1.WsConnection(connName, ws, this.emitter);
        if (this.heartbeatInterval) {
            conn.setHeartbeatInterval(this.heartbeatInterval);
        }
        conn.on('bye', reason => {
            this.log.debug('Peer intentionally closing:', reason);
            if (this.setClosing()) {
                this.emitter.emit('__close', reason);
            }
        });
        conn.on('close', (code, reason) => {
            this.log.debug('Peer closed:', reason);
            this.dropConnection(conn, `Peer closed: ${code} ${reason}`);
        });
        conn.on('error', error => {
            this.dropConnection(conn, `Connection error: ${node_util_1.default.inspect(error)}`);
        });
        return conn;
    }
    setClosing() {
        if (this.closing) {
            return false;
        }
        this.closing = true;
        this.newEpoch();
        this.queue.forEach(item => {
            item.deferred?.reject(new Error('WsChannel has been closed.'));
        });
        return true;
    }
    dropConnection(conn, reason) {
        if (this.closing) {
            this.log.debug('Clean up:', reason);
            return;
        }
        if (this.connection !== conn) {
            this.log.debug(`Previous connection closed: ${reason}`);
            return;
        }
        this.log.warning('Connection closed unexpectedly:', reason);
        this.newEpoch();
        this.emitter.emit('__lost');
        if (!this.terminateTimer) {
            this.terminateTimer = setTimeout(() => {
                if (!this.closing) {
                    this.terminate('have not reconnected in 30s');
                }
            }, terminateTimeout);
        }
    }
}
exports.WsChannel = WsChannel;
let defaultHeartbeatInterval = 5000;
let terminateTimeout = 30000;
var UnitTestHelper;
(function (UnitTestHelper) {
    function setHeartbeatInterval(ms) {
        defaultHeartbeatInterval = ms;
    }
    UnitTestHelper.setHeartbeatInterval = setHeartbeatInterval;
    function setTerminateTimeout(ms) {
        terminateTimeout = ms;
    }
    UnitTestHelper.setTerminateTimeout = setTerminateTimeout;
    function reset() {
        defaultHeartbeatInterval = 5000;
        terminateTimeout = 30000;
    }
    UnitTestHelper.reset = reset;
})(UnitTestHelper = exports.UnitTestHelper || (exports.UnitTestHelper = {}));
