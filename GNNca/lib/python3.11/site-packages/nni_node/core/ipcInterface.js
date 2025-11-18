"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.encodeCommand = exports.createDispatcherInterface = void 0;
const tuner_command_channel_1 = require("./tuner_command_channel");
let tunerDisabled = false;
async function createDispatcherInterface() {
    if (!tunerDisabled) {
        return (0, tuner_command_channel_1.getTunerServer)();
    }
    else {
        return new DummyIpcInterface();
    }
}
exports.createDispatcherInterface = createDispatcherInterface;
function encodeCommand(commandType, content) {
    const contentBuffer = Buffer.from(content);
    const contentLengthBuffer = Buffer.from(contentBuffer.length.toString().padStart(14, '0'));
    return Buffer.concat([Buffer.from(commandType), contentLengthBuffer, contentBuffer]);
}
exports.encodeCommand = encodeCommand;
class DummyIpcInterface {
    async init() { }
    sendCommand(_commandType, _content) { }
    onCommand(_listener) { }
    onError(_listener) { }
}
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function disableTuner() {
        tunerDisabled = true;
    }
    UnitTestHelpers.disableTuner = disableTuner;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
