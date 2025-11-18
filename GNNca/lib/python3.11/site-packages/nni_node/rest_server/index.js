"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.RestServer = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const path_1 = __importDefault(require("path"));
const express_1 = __importStar(require("express"));
const http_proxy_1 = __importDefault(require("http-proxy"));
const deferred_1 = require("common/deferred");
const globals_1 = __importDefault(require("common/globals"));
const log_1 = require("common/log");
const logger = (0, log_1.getLogger)('RestServer');
class RestServer {
    port;
    urlPrefix;
    server = null;
    constructor(port, urlPrefix) {
        (0, strict_1.default)(!urlPrefix.startsWith('/') && !urlPrefix.endsWith('/'));
        this.port = port;
        this.urlPrefix = urlPrefix;
        globals_1.default.shutdown.register('RestServer', this.shutdown.bind(this));
    }
    start() {
        logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);
        const app = globals_1.default.rest.getExpressApp();
        app.use('/' + this.urlPrefix, mainRouter());
        app.use('/' + this.urlPrefix, fallbackRouter());
        app.all('*', (_req, res) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);
        const deferred = new deferred_1.Deferred();
        this.server.on('listening', () => {
            if (this.port === 0) {
                this.port = this.server.address().port;
            }
            logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', (error) => { globals_1.default.shutdown.criticalError('RestServer', error); });
        return deferred.promise;
    }
    shutdown() {
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
        }, 5000);
        return deferred.promise;
    }
}
exports.RestServer = RestServer;
function mainRouter() {
    const router = globals_1.default.rest.getExpressRouter();
    const logRouter = (0, express_1.Router)();
    logRouter.get('*', express_1.default.static(globals_1.default.paths.logDirectory));
    router.use('/logs', logRouter);
    router.use('/netron', netronProxy());
    return router;
}
function fallbackRouter() {
    const router = (0, express_1.Router)();
    router.get('/api/v1/nni/check-status', (_req, res) => { res.send('INITIALIZING'); });
    router.get('*', express_1.default.static(webuiPath));
    router.get('*', (_req, res) => { res.sendFile(path_1.default.join(webuiPath, 'index.html')); });
    router.all('*', (_req, res) => { res.status(404).send('Not Found'); });
    return router;
}
function netronProxy() {
    const router = (0, express_1.Router)();
    const proxy = http_proxy_1.default.createProxyServer();
    router.all('*', (req, res) => {
        delete req.headers.host;
        proxy.web(req, res, { changeOrigin: true, target: netronUrl });
    });
    return router;
}
let webuiPath = path_1.default.resolve('static');
let netronUrl = 'https://netron.app';
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function getPort(server) {
        return server.port;
    }
    UnitTestHelpers.getPort = getPort;
    function setWebuiPath(mockPath) {
        webuiPath = path_1.default.resolve(mockPath);
    }
    UnitTestHelpers.setWebuiPath = setWebuiPath;
    function setNetronUrl(mockUrl) {
        netronUrl = mockUrl;
    }
    UnitTestHelpers.setNetronUrl = setNetronUrl;
    function reset() {
        webuiPath = path_1.default.resolve('static');
        netronUrl = 'https://netron.app';
    }
    UnitTestHelpers.reset = reset;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
