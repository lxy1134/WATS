"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createTarball = exports.getTarballPath = void 0;
const promises_1 = __importDefault(require("node:fs/promises"));
const node_path_1 = __importDefault(require("node:path"));
const ignore_1 = __importDefault(require("ignore"));
const tar_1 = __importDefault(require("tar"));
const globals_1 = require("common/globals");
const log_1 = require("common/log");
const logger = (0, log_1.getLogger)('common.tarball');
function getTarballPath(tarName) {
    const tarDir = node_path_1.default.join(globals_1.globals.paths.experimentRoot, 'tarball');
    return node_path_1.default.join(tarDir, `${tarName}.tgz`);
}
exports.getTarballPath = getTarballPath;
async function createTarball(tarName, sourcePath) {
    const fileList = [];
    let ignorePatterns;
    try {
        ignorePatterns = await promises_1.default.readFile(node_path_1.default.join(sourcePath, '.nniignore'), { encoding: 'utf8' });
    }
    catch { }
    const ig = ignorePatterns ? (0, ignore_1.default)().add(ignorePatterns) : undefined;
    let countNum = 0;
    let countSize = 0;
    for await (const [file, stats] of walk(sourcePath, '', ig)) {
        if (stats.isSymbolicLink()) {
            logger.warning(`${sourcePath} contains a symlink ${file}. It will be uploaded as is and might not work`);
        }
        fileList.push(file);
        countNum += 1;
        countSize += stats.size;
        if (countNum > 2000) {
            logger.error(`Failed to pack ${sourcePath}: too many files`);
            throw new Error(`${sourcePath} contains too many files (more than 2000)`);
        }
        if (countSize > 300 * 1024 * 1024) {
            logger.error(`Failed to pack ${sourcePath}: too large`);
            throw new Error(`${sourcePath} is too large (more than 300MB)`);
        }
    }
    const tarPath = getTarballPath(tarName);
    await promises_1.default.mkdir(node_path_1.default.dirname(tarPath), { recursive: true });
    const opts = {
        gzip: true,
        file: tarPath,
        cwd: sourcePath,
        portable: true,
    };
    await tar_1.default.create(opts, fileList);
    return tarPath;
}
exports.createTarball = createTarball;
async function* walk(root, relDir, ig) {
    const dir = node_path_1.default.join(root, relDir);
    const entries = await promises_1.default.readdir(dir);
    for (const entry of entries) {
        const stats = await promises_1.default.lstat(node_path_1.default.join(dir, entry));
        const relEntry = node_path_1.default.join(relDir, entry);
        if (ig && ig.ignores(relEntry + (stats.isDirectory() ? node_path_1.default.sep : ''))) {
            continue;
        }
        if (stats.isDirectory()) {
            yield* walk(root, relEntry, ig);
        }
        else {
            yield [relEntry, stats];
        }
    }
}
