"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WsChannelClient = void 0;
const promises_1 = require("node:timers/promises");
const ws_1 = require("ws");
const log_1 = require("common/log");
const channel_1 = require("./channel");
const maxPayload = 1024 * 1024 * 1024;
class WsChannelClient extends channel_1.WsChannel {
    logger;
    reconnecting = false;
    url;
    constructor(name, url) {
        super(name);
        this.logger = (0, log_1.getLogger)(`WsChannelClient.${name}`);
        this.url = url;
        this.onLost(this.reconnect.bind(this));
    }
    async connect() {
        this.logger.debug('Connecting to', this.url);
        const ws = new ws_1.WebSocket(this.url, { maxPayload });
        await this.setConnection(ws, true),
            this.logger.debug('Connected');
    }
    async disconnect(reason) {
        this.close(reason ?? 'client intentionally disconnect');
    }
    async reconnect() {
        if (this.reconnecting) {
            return;
        }
        this.reconnecting = true;
        this.logger.warning('Connection lost. Try to reconnect');
        for (let i = 0; i <= 5; i++) {
            if (i > 0) {
                this.logger.warning(`Wait ${i}s before next try`);
                await (0, promises_1.setTimeout)(i * 1000);
            }
            try {
                await this.connect();
                this.logger.info('Reconnect success');
                this.reconnecting = false;
                return;
            }
            catch (error) {
                this.logger.warning('Reconnect failed:', error);
            }
        }
        this.logger.error('Conenction lost. Cannot reconnect');
        this.emitter.emit('__error', new Error('Connection lost'));
    }
}
exports.WsChannelClient = WsChannelClient;
