"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.IocShim = void 0;
const strict_1 = __importDefault(require("node:assert/strict"));
class IocShimClass {
    singletons = new Map();
    snapshots = new Map();
    bind(keyClass, valueClass) {
        const key = keyClass.name;
        strict_1.default.ok(!this.singletons.has(key));
        this.singletons.set(key, new valueClass());
    }
    bindInstance(keyClass, value) {
        const key = keyClass.name;
        strict_1.default.ok(!this.singletons.has(key));
        this.singletons.set(key, value);
    }
    get(keyClass) {
        const key = keyClass.name;
        strict_1.default.ok(this.singletons.has(key));
        return this.singletons.get(key);
    }
    snapshot(keyClass) {
        const key = keyClass.name;
        const value = this.singletons.get(key);
        this.snapshots.set(key, value);
    }
    restore(keyClass) {
        const key = keyClass.name;
        const value = this.snapshots.get(key);
        this.singletons.set(key, value);
    }
    clear() {
        this.singletons.clear();
    }
}
exports.IocShim = new IocShimClass();
