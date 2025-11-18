"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.getTunerServer = void 0;
const node_events_1 = require("node:events");
const websocket_1 = require("common/command_channel/websocket");
const deferred_1 = require("common/deferred");
const log_1 = require("common/log");
function getTunerServer() {
    return server;
}
exports.getTunerServer = getTunerServer;
const logger = (0, log_1.getLogger)('tuner_command_channel');
class TunerServer {
    channel;
    connect = new deferred_1.Deferred();
    emitter = new node_events_1.EventEmitter();
    server;
    constructor() {
        this.server = new websocket_1.WsChannelServer('tuner', 'tuner');
        this.server.onConnection((_channelId, channel) => {
            this.channel = channel;
            this.channel.onError(error => {
                this.emitter.emit('error', error);
            });
            this.channel.onReceive(command => {
                if (command.type === 'ER') {
                    this.emitter.emit('error', new Error(command.content));
                }
                else {
                    this.emitter.emit('command', command.type, command.content ?? '');
                }
            });
            this.connect.resolve();
        });
        this.server.start();
    }
    init() {
        if (this.connect.settled) {
            logger.debug('Initialized.');
            return Promise.resolve();
        }
        else {
            logger.debug('Waiting connection...');
            setTimeout(() => {
                if (!this.connect.settled) {
                    const msg = 'Tuner did not connect in 10 seconds. Please check tuner (dispatcher) log.';
                    this.connect.reject(new Error('tuner_command_channel: ' + msg));
                }
            }, 10000);
            return this.connect.promise;
        }
    }
    async stop() {
        await this.server.shutdown();
    }
    sendCommand(commandType, content) {
        if (commandType === 'PI') {
            return;
        }
        if (this.channel.getBufferedAmount() > 1000) {
            logger.warning('Sending too fast! Try to reduce the frequency of intermediate results.');
        }
        this.channel.send({ type: commandType, content });
        if (commandType === 'TE') {
            this.channel.close('TE command');
            this.server.shutdown();
        }
    }
    onCommand(listener) {
        this.emitter.on('command', listener);
    }
    onError(listener) {
        this.emitter.on('error', listener);
    }
}
let server = new TunerServer();
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function reset() {
        server = new TunerServer();
    }
    UnitTestHelpers.reset = reset;
    async function stop() {
        await server.stop();
    }
    UnitTestHelpers.stop = stop;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
