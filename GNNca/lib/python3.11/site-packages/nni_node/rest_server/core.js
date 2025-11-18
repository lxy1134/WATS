"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RestServerCore = void 0;
const strict_1 = __importDefault(require("node:assert/strict"));
const deferred_1 = require("common/deferred");
const globals_1 = require("common/globals");
const log_1 = require("common/log");
const logger = (0, log_1.getLogger)('RestServerCore');
class RestServerCore {
    port;
    urlPrefix;
    server = null;
    constructor(port, urlPrefix) {
        this.port = port ?? 0;
        this.urlPrefix = urlPrefix ?? '';
        (0, strict_1.default)(!this.urlPrefix.startsWith('/') && !this.urlPrefix.endsWith('/'));
        globals_1.globals.shutdown.register('RestServerCore', this.shutdown.bind(this));
    }
    start() {
        logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);
        const app = globals_1.globals.rest.getExpressApp();
        app.use('/' + this.urlPrefix, globals_1.globals.rest.getExpressRouter());
        app.all('/' + this.urlPrefix, (_req, res) => { res.status(404).send('Not Found'); });
        app.all('*', (_req, res) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);
        const deferred = new deferred_1.Deferred();
        this.server.on('listening', () => {
            if (this.port === 0) {
                this.port = this.server.address().port;
                globals_1.globals.args.port = this.port;
            }
            logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', error => { globals_1.globals.shutdown.criticalError('RestServer', error); });
        return deferred.promise;
    }
    shutdown(timeoutMilliseconds) {
        logger.info('Stopping REST server.');
        if (this.server === null) {
            logger.warning('REST server is not running.');
            return Promise.resolve();
        }
        const deferred = new deferred_1.Deferred();
        this.server.close(() => {
            logger.info('REST server stopped.');
            deferred.resolve();
        });
        setTimeout(() => {
            if (!deferred.settled) {
                logger.debug('Killing connections');
                this.server?.closeAllConnections();
            }
        }, timeoutMilliseconds ?? 5000);
        return deferred.promise;
    }
}
exports.RestServerCore = RestServerCore;
