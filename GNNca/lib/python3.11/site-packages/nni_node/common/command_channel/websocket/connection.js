"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.WsConnection = void 0;
const node_events_1 = require("node:events");
const node_util_1 = __importDefault(require("node:util"));
const log_1 = require("common/log");
class WsConnection extends node_events_1.EventEmitter {
    closing = false;
    commandEmitter;
    heartbeatTimer = null;
    log;
    missingPongs = 0;
    ws;
    constructor(name, ws, commandEmitter) {
        super();
        this.log = (0, log_1.getLogger)(`WsConnection.${name}`);
        this.ws = ws;
        this.commandEmitter = commandEmitter;
        ws.on('close', this.handleClose.bind(this));
        ws.on('error', this.handleError.bind(this));
        ws.on('message', this.handleMessage.bind(this));
        ws.on('pong', this.handlePong.bind(this));
    }
    setHeartbeatInterval(interval) {
        if (this.heartbeatTimer) {
            clearTimeout(this.heartbeatTimer);
        }
        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), interval);
    }
    async close(reason) {
        if (this.closing) {
            this.log.debug('Close again:', reason);
            return;
        }
        this.log.debug('Close connection:', reason);
        this.closing = true;
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
        try {
            await this.sendAsync({ type: '_bye_', reason });
        }
        catch (error) {
            this.log.error('Failed to send bye:', error);
        }
        try {
            this.ws.close(4000, reason);
            return;
        }
        catch (error) {
            this.log.error('Failed to close socket:', error);
        }
        try {
            this.ws.terminate();
        }
        catch (error) {
            this.log.debug('Failed to terminate socket:', error);
        }
    }
    terminate(reason) {
        this.log.debug('Terminate connection:', reason);
        this.closing = true;
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
        try {
            this.ws.close(4001, reason);
            return;
        }
        catch (error) {
            this.log.debug('Failed to close socket:', error);
        }
        try {
            this.ws.terminate();
        }
        catch (error) {
            this.log.debug('Failed to terminate socket:', error);
        }
    }
    send(command) {
        this.log.trace('Send command', command);
        this.ws.send(JSON.stringify(command));
    }
    sendAsync(command) {
        this.log.trace('(async) Send command', command);
        const send = node_util_1.default.promisify(this.ws.send.bind(this.ws));
        return send(JSON.stringify(command));
    }
    handleClose(code, reason) {
        if (this.closing) {
            this.log.debug('Connection closed');
        }
        else {
            this.log.debug('Connection closed by peer:', code, String(reason));
            this.emit('close', code, String(reason));
        }
    }
    handleError(error) {
        if (this.closing) {
            this.log.warning('Error after closing:', error);
        }
        else {
            this.log.error('Connection error:', error);
            this.emit('error', error);
        }
    }
    handleMessage(data, _isBinary) {
        const s = String(data);
        if (this.closing) {
            this.log.warning('Received message after closing:', s);
            return;
        }
        this.log.trace('Receive command', s);
        const command = JSON.parse(s);
        if (command.type === '_nop_') {
            return;
        }
        if (command.type === '_bye_') {
            this.log.debug('Intentionally close connection:', s);
            this.closing = true;
            this.emit('bye', command.reason);
            return;
        }
        const hasReceiveListener = this.commandEmitter.emit('__receive', command);
        const hasCommandListener = this.commandEmitter.emit(command.type, command);
        if (!hasReceiveListener && !hasCommandListener) {
            this.log.warning('No listener for command', s);
        }
    }
    handlePong() {
        this.log.trace('Receive pong');
        this.missingPongs = 0;
    }
    heartbeat() {
        if (this.missingPongs > 0) {
            this.log.warning('Missing pong');
        }
        if (this.missingPongs > 3) {
            this.sendAsync({ type: '_nop_' }).then(() => {
                this.missingPongs = 0;
            }).catch(error => {
                this.log.error('Failed sending command. Drop connection:', error);
                this.terminate(`peer lost responsive: ${node_util_1.default.inspect(error)}`);
            });
        }
        this.missingPongs += 1;
        this.log.trace('Send ping');
        this.ws.ping();
    }
}
exports.WsConnection = WsConnection;
