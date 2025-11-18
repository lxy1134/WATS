"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.WsChannelServer = void 0;
const events_1 = require("events");
const deferred_1 = require("common/deferred");
const globals_1 = __importDefault(require("common/globals"));
const log_1 = require("common/log");
const channel_1 = require("./channel");
class WsChannelServer extends events_1.EventEmitter {
    channels = new Map();
    log;
    path;
    receiveCallbacks = [];
    constructor(name, urlPath) {
        super();
        this.log = (0, log_1.getLogger)(`WsChannelServer.${name}`);
        this.path = urlPath;
    }
    async start() {
        const channelPath = globals_1.default.rest.urlJoin(this.path, ':channel');
        globals_1.default.rest.registerWebSocketHandler(this.path, (ws, _req) => {
            this.handleConnection('__default__', ws);
        });
        globals_1.default.rest.registerWebSocketHandler(channelPath, (ws, req) => {
            this.handleConnection(req.params['channel'], ws);
        });
        this.log.debug('Start listening', channelPath);
    }
    shutdown() {
        const deferred = new deferred_1.Deferred();
        this.channels.forEach((channel, channelId) => {
            channel.onClose(_reason => {
                this.channels.delete(channelId);
                if (this.channels.size === 0) {
                    deferred.resolve();
                }
            });
            channel.close('shutdown');
        });
        setTimeout(() => {
            this.log.debug('Shutdown timeout. Stop waiting following channels:', Array.from(this.channels.keys()));
            deferred.resolve();
        }, 5000);
        return deferred.promise;
    }
    getChannelUrl(channelId, ip) {
        return globals_1.default.rest.getFullUrl('ws', ip ?? 'localhost', this.path, channelId);
    }
    send(channelId, command) {
        const channel = this.channels.get(channelId);
        if (channel) {
            channel.send(command);
        }
        else {
            this.log.error(`Channel ${channelId} is not available`);
        }
    }
    onReceive(callback) {
        this.receiveCallbacks.push(callback);
        for (const [channelId, channel] of this.channels) {
            channel.onReceive(command => { callback(channelId, command); });
        }
    }
    onConnection(callback) {
        this.on('connection', callback);
    }
    handleConnection(channelId, ws) {
        this.log.debug('Incoming connection', channelId);
        if (this.channels.has(channelId)) {
            this.log.warning(`Channel ${channelId} reconnecting, drop previous connection`);
            this.channels.get(channelId).setConnection(ws, false);
            return;
        }
        const channel = new channel_1.WsChannel(channelId);
        this.channels.set(channelId, channel);
        channel.onClose(reason => {
            this.log.debug(`Connection ${channelId} closed:`, reason);
            this.channels.delete(channelId);
        });
        channel.onError(error => {
            this.log.error(`Connection ${channelId} error:`, error);
            this.channels.delete(channelId);
        });
        for (const cb of this.receiveCallbacks) {
            channel.onReceive(command => { cb(channelId, command); });
        }
        channel.enableHeartbeat();
        channel.setConnection(ws, false);
        this.emit('connection', channelId, channel);
    }
}
exports.WsChannelServer = WsChannelServer;
