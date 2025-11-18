"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DefaultMap = void 0;
class DefaultMap extends Map {
    defaultFactory;
    constructor(defaultFactory) {
        super();
        this.defaultFactory = defaultFactory;
    }
    get(key) {
        const value = super.get(key);
        if (value !== undefined) {
            return value;
        }
        const defaultValue = this.defaultFactory();
        this.set(key, defaultValue);
        return defaultValue;
    }
}
exports.DefaultMap = DefaultMap;
