"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
require("app-module-path/register");
const globals_1 = require("common/globals");
(0, globals_1.initGlobals)();
const datastore_1 = require("common/datastore");
const ioc_shim_1 = require("common/ioc_shim");
const log_1 = require("common/log");
const manager_1 = require("common/manager");
const tensorboardManager_1 = require("common/tensorboardManager");
const nniDataStore_1 = require("core/nniDataStore");
const nnimanager_1 = require("core/nnimanager");
const sqlDatabase_1 = require("core/sqlDatabase");
const experiments_manager_1 = require("extensions/experiments_manager");
const nniTensorboardManager_1 = require("extensions/nniTensorboardManager");
const rest_server_1 = require("rest_server");
const restHandler_1 = require("rest_server/restHandler");
const logger = (0, log_1.getLogger)('main');
async function start() {
    logger.info('Start NNI manager');
    const restServer = new rest_server_1.RestServer(globals_1.globals.args.port, globals_1.globals.args.urlPrefix);
    await restServer.start();
    ioc_shim_1.IocShim.bind(datastore_1.Database, sqlDatabase_1.SqlDB);
    ioc_shim_1.IocShim.bind(datastore_1.DataStore, nniDataStore_1.NNIDataStore);
    ioc_shim_1.IocShim.bind(manager_1.Manager, nnimanager_1.NNIManager);
    ioc_shim_1.IocShim.bind(tensorboardManager_1.TensorboardManager, nniTensorboardManager_1.NNITensorboardManager);
    const ds = ioc_shim_1.IocShim.get(datastore_1.DataStore);
    await ds.init();
    globals_1.globals.rest.registerExpressRouter('/api/v1/nni', (0, restHandler_1.createRestHandler)());
    (0, experiments_manager_1.initExperimentsManager)();
    globals_1.globals.shutdown.notifyInitializeComplete();
}
process.on('SIGTERM', () => { globals_1.globals.shutdown.initiate('SIGTERM'); });
process.on('SIGBREAK', () => { globals_1.globals.shutdown.initiate('SIGBREAK'); });
process.on('SIGINT', () => { globals_1.globals.shutdown.initiate('SIGINT'); });
start().then(() => {
    logger.debug('start() returned.');
}).catch((error) => {
    try {
        logger.error('Failed to start:', error);
    }
    catch (loggerError) {
        console.error('Failed to start:', error);
        console.error('Seems logger is faulty:', loggerError);
    }
    process.exit(1);
});
