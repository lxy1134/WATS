"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createRestHandler = void 0;
const express_1 = require("express");
const path_1 = __importDefault(require("path"));
const ioc_shim_1 = require("common/ioc_shim");
const datastore_1 = require("../common/datastore");
const errors_1 = require("../common/errors");
const experimentStartupInfo_1 = require("../common/experimentStartupInfo");
const globals_1 = __importDefault(require("common/globals"));
const log_1 = require("../common/log");
const manager_1 = require("../common/manager");
const experiments_manager_1 = require("extensions/experiments_manager");
const tensorboardManager_1 = require("../common/tensorboardManager");
const utils_1 = require("../common/utils");
class NNIRestHandler {
    nniManager;
    tensorboardManager;
    log;
    constructor() {
        this.nniManager = ioc_shim_1.IocShim.get(manager_1.Manager);
        this.tensorboardManager = ioc_shim_1.IocShim.get(tensorboardManager_1.TensorboardManager);
        this.log = (0, log_1.getLogger)('NNIRestHandler');
    }
    createRestHandler() {
        const router = (0, express_1.Router)();
        router.use((req, res, next) => {
            this.log.debug(`${req.method}: ${req.url}: body:`, req.body);
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
            res.header('Access-Control-Allow-Methods', 'PUT,POST,GET,DELETE,OPTIONS');
            res.setHeader('Content-Type', 'application/json');
            next();
        });
        this.version(router);
        this.checkStatus(router);
        this.getExperimentProfile(router);
        this.getExperimentMetadata(router);
        this.updateExperimentProfile(router);
        this.importData(router);
        this.getImportedData(router);
        this.startExperiment(router);
        this.getTrialJobStatistics(router);
        this.setClusterMetaData(router);
        this.listTrialJobs(router);
        this.getTrialJob(router);
        this.addTrialJob(router);
        this.cancelTrialJob(router);
        this.getMetricData(router);
        this.getMetricDataByRange(router);
        this.getLatestMetricData(router);
        this.getTrialFile(router);
        this.exportData(router);
        this.getExperimentsInfo(router);
        this.startTensorboardTask(router);
        this.getTensorboardTask(router);
        this.updateTensorboardTask(router);
        this.stopTensorboardTask(router);
        this.stopAllTensorboardTask(router);
        this.listTensorboardTask(router);
        this.stop(router);
        router.use((err, _req, res, _next) => {
            if (err.isBoom) {
                this.log.error(err.output.payload);
                return res.status(err.output.statusCode).json(err.output.payload);
            }
        });
        return router;
    }
    handleError(err, res, isFatal = false, errorCode = 500) {
        if (err instanceof errors_1.NNIError && err.name === errors_1.NNIErrorNames.NOT_FOUND) {
            res.status(404);
        }
        else {
            res.status(errorCode);
        }
        res.send({
            error: err.message
        });
        if (isFatal) {
            this.log.critical(err);
            process.exit(1);
        }
        else {
            this.log.error(err);
        }
    }
    version(router) {
        router.get('/version', async (_req, res) => {
            const version = await (0, utils_1.getVersion)();
            res.send(version);
        });
    }
    checkStatus(router) {
        router.get('/check-status', (_req, res) => {
            const ds = ioc_shim_1.IocShim.get(datastore_1.DataStore);
            ds.init().then(() => {
                res.send(this.nniManager.getStatus());
            }).catch(async (err) => {
                this.handleError(err, res);
                this.log.error(err.message);
                this.log.error(`Datastore initialize failed, stopping rest server...`);
                globals_1.default.shutdown.criticalError('RestHandler', err);
            });
        });
    }
    getExperimentProfile(router) {
        router.get('/experiment', (_req, res) => {
            this.nniManager.getExperimentProfile().then((profile) => {
                res.send(profile);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    updateExperimentProfile(router) {
        router.put('/experiment', (req, res) => {
            this.nniManager.updateExperimentProfile(req.body, req.query['update_type']).then(() => {
                res.send();
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    importData(router) {
        router.post('/experiment/import-data', (req, res) => {
            this.nniManager.importData(JSON.stringify(req.body)).then(() => {
                res.send();
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getImportedData(router) {
        router.get('/experiment/imported-data', (_req, res) => {
            this.nniManager.getImportedData().then((importedData) => {
                res.send(JSON.stringify(importedData));
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    startExperiment(router) {
        router.post('/experiment', (req, res) => {
            if ((0, experimentStartupInfo_1.isNewExperiment)()) {
                this.nniManager.startExperiment(req.body).then((eid) => {
                    res.send({
                        experiment_id: eid
                    });
                }).catch((err) => {
                    this.handleError(err, res);
                });
            }
            else {
                this.nniManager.resumeExperiment((0, experimentStartupInfo_1.isReadonly)()).then(() => {
                    res.send();
                }).catch((err) => {
                    this.handleError(err, res);
                });
            }
        });
    }
    getTrialJobStatistics(router) {
        router.get('/job-statistics', (_req, res) => {
            this.nniManager.getTrialJobStatistics().then((statistics) => {
                res.send(statistics);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    setClusterMetaData(router) {
        router.put('/experiment/cluster-metadata', async (req, res) => {
            const metadata = req.body;
            const keys = Object.keys(metadata);
            try {
                for (const key of keys) {
                    await this.nniManager.setClusterMetadata(key, JSON.stringify(metadata[key]));
                }
                res.send();
            }
            catch (err) {
                this.handleError(errors_1.NNIError.FromError(err), res, true);
            }
        });
    }
    listTrialJobs(router) {
        router.get('/trial-jobs', (req, res) => {
            this.nniManager.listTrialJobs(req.query['status']).then((jobInfos) => {
                jobInfos.forEach((trialJob) => {
                    this.setErrorPathForFailedJob(trialJob);
                });
                res.send(jobInfos);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getTrialJob(router) {
        router.get('/trial-jobs/:id', (req, res) => {
            this.nniManager.getTrialJob(req.params['id']).then((jobDetail) => {
                const jobInfo = this.setErrorPathForFailedJob(jobDetail);
                res.send(jobInfo);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    addTrialJob(router) {
        router.post('/trial-jobs', async (req, res) => {
            this.nniManager.addCustomizedTrialJob(JSON.stringify(req.body)).then((sequenceId) => {
                res.send({ sequenceId });
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    cancelTrialJob(router) {
        router.delete('/trial-jobs/:id', async (req, res) => {
            this.nniManager.cancelTrialJobByUser(req.params['id']).then(() => {
                res.send();
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getMetricData(router) {
        router.get('/metric-data/:job_id*?', async (req, res) => {
            this.nniManager.getMetricData(req.params['job_id'], req.query['type']).then((metricsData) => {
                res.send(metricsData);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getMetricDataByRange(router) {
        router.get('/metric-data-range/:min_seq_id/:max_seq_id', async (req, res) => {
            const minSeqId = Number(req.params['min_seq_id']);
            const maxSeqId = Number(req.params['max_seq_id']);
            this.nniManager.getMetricDataByRange(minSeqId, maxSeqId).then((metricsData) => {
                res.send(metricsData);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getLatestMetricData(router) {
        router.get('/metric-data-latest/', async (_req, res) => {
            this.nniManager.getLatestMetricData().then((metricsData) => {
                res.send(metricsData);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getTrialFile(router) {
        router.get('/trial-file/:id/:filename', async (req, res) => {
            const filename = req.params['filename'];
            this.nniManager.getTrialFile(req.params['id'], filename).then((content) => {
                const contentType = content instanceof Buffer ? 'application/octet-stream' : 'text/plain';
                res.header('Content-Type', contentType);
                if (content === '') {
                    content = `${filename} is empty.`;
                }
                res.send(content);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    exportData(router) {
        router.get('/export-data', (_req, res) => {
            this.nniManager.exportData().then((exportedData) => {
                res.send(exportedData);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getExperimentMetadata(router) {
        router.get('/experiment-metadata', (_req, res) => {
            Promise.all([
                this.nniManager.getExperimentProfile(),
                (0, experiments_manager_1.getExperimentsManager)().getExperimentsInfo()
            ]).then(([profile, experimentInfo]) => {
                for (const info of experimentInfo) {
                    if (info.id === profile.id) {
                        res.send(info);
                        break;
                    }
                }
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    getExperimentsInfo(router) {
        router.get('/experiments-info', (_req, res) => {
            (0, experiments_manager_1.getExperimentsManager)().getExperimentsInfo().then((experimentInfo) => {
                res.send(JSON.stringify(experimentInfo));
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    startTensorboardTask(router) {
        router.post('/tensorboard', (req, res) => {
            this.tensorboardManager.startTensorboardTask(req.body).then((taskDetail) => {
                this.log.info(taskDetail);
                res.send(Object.assign({}, taskDetail));
            }).catch((err) => {
                this.handleError(err, res, false, 400);
            });
        });
    }
    getTensorboardTask(router) {
        router.get('/tensorboard/:id', (req, res) => {
            this.tensorboardManager.getTensorboardTask(req.params['id']).then((taskDetail) => {
                res.send(Object.assign({}, taskDetail));
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    updateTensorboardTask(router) {
        router.put('/tensorboard/:id', (req, res) => {
            this.tensorboardManager.updateTensorboardTask(req.params['id']).then((taskDetail) => {
                res.send(Object.assign({}, taskDetail));
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    stopTensorboardTask(router) {
        router.delete('/tensorboard/:id', (req, res) => {
            this.tensorboardManager.stopTensorboardTask(req.params['id']).then((taskDetail) => {
                res.send(Object.assign({}, taskDetail));
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    stopAllTensorboardTask(router) {
        router.delete('/tensorboard-tasks', (_req, res) => {
            this.tensorboardManager.stopAllTensorboardTask().then(() => {
                res.send();
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    listTensorboardTask(router) {
        router.get('/tensorboard-tasks', (_req, res) => {
            this.tensorboardManager.listTensorboardTasks().then((taskDetails) => {
                res.send(taskDetails);
            }).catch((err) => {
                this.handleError(err, res);
            });
        });
    }
    stop(router) {
        router.delete('/experiment', (_req, res) => {
            res.send();
            globals_1.default.shutdown.initiate('REST request');
        });
    }
    setErrorPathForFailedJob(jobInfo) {
        if (jobInfo === undefined || jobInfo.status !== 'FAILED' || jobInfo.logPath === undefined) {
            return jobInfo;
        }
        jobInfo.stderrPath = path_1.default.join(jobInfo.logPath, 'stderr');
        return jobInfo;
    }
}
function createRestHandler() {
    return new NNIRestHandler().createRestHandler();
}
exports.createRestHandler = createRestHandler;
