"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createScriptFile = exports.powershellString = exports.bashString = exports.shellString = void 0;
const promises_1 = __importDefault(require("fs/promises"));
const singleQuote = "'";
const doubleQuote = '"';
const backtick = '`';
const backslash = '\\';
const doubleBacktick = '``';
const doubleBackslash = '\\\\';
const newline = '\n';
function shellString(str) {
    return process.platform === 'win32' ? powershellString(str) : bashString(str);
}
exports.shellString = shellString;
function bashString(str) {
    if (str.includes(singleQuote) || str.includes(newline)) {
        str = str.replaceAll(backslash, doubleBackslash);
        str = str.replaceAll(singleQuote, backslash + singleQuote);
        str = str.replaceAll(newline, backslash + 'n');
        return '$' + singleQuote + str + singleQuote;
    }
    else {
        return singleQuote + str + singleQuote;
    }
}
exports.bashString = bashString;
function powershellString(str) {
    if (str.includes(newline)) {
        str = str.replaceAll(backtick, doubleBacktick);
        str = str.replaceAll(doubleQuote, backtick + doubleQuote);
        str = str.replaceAll(newline, backtick + 'n');
        str = str.replaceAll('$', backtick + '$');
        return doubleQuote + str + doubleQuote;
    }
    else {
        str = str.replaceAll(singleQuote, singleQuote + singleQuote);
        return singleQuote + str + singleQuote;
    }
}
exports.powershellString = powershellString;
function createScriptFile(path, content) {
    if (path.endsWith('.ps1') && !/^[\x00-\x7F]*$/.test(content)) {
        content = '\uFEFF' + content;
    }
    return promises_1.default.writeFile(path, content, { mode: 0o777 });
}
exports.createScriptFile = createScriptFile;
