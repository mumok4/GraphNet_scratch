const BlockType = require('../../extension-support/block-type');
const ArgumentType = require('../../extension-support/argument-type');
const Cast = require('../../util/cast');
const bindAll = require('lodash.bindall');
const formatMessage = require('format-message');

class Scratch3NeuralNet {
constructor(runtime) {
this.runtime = runtime;

this.isModelLoaded = false;
    this._modelInputWidth = 0;
    this._modelInputHeight = 0;
    this.labels = [];
    this.lastProbabilities = [];

    this._canvas = document.createElement('canvas');
    this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
    
    this.worker = new Worker(new URL('./tf-worker.js', import.meta.url));
    this.pendingPromises = new Map();
    this.nextPromiseId = 0;

    bindAll(this, [
        'handleWorkerMessage',
        'loadModel',
        'loadLabels',
        'predictImage',
        'getClassName',
        'getConfidence',
        'modelInputWidth',
        'modelInputHeight',
        'isLoaded'
    ]);

    this.worker.onmessage = this.handleWorkerMessage;
    this.worker.onerror = (err) => {
        alert(`Критическая ошибка в фоновом процессе: ${err.message}`);
    };
}

getInfo() {
    return {
        id: 'scratch3neuralnet',
        name: 'Нейросеть',
        blocks: [
            {
                opcode: 'loadModel',
                blockType: BlockType.COMMAND,
                text: '📁 Загрузить модель (.json и .bin)',
            },
            {
                opcode: 'loadLabels',
                blockType: BlockType.COMMAND,
                text: '🏷️ Загрузить метки классов (.txt)',
            },
            '---',
            {
                opcode: 'predictImage',
                blockType: BlockType.REPORTER,
                text: '🔮 распознать с [IMAGE_SOURCE]',
                arguments: {
                    IMAGE_SOURCE: {
                        type: ArgumentType.STRING,
                        menu: 'imageSourceMenu',
                        defaultValue: 'costume'
                    }
                }
            },
            '---',
            {
                opcode: 'getClassName',
                blockType: BlockType.REPORTER,
                text: '🏷️ Имя класса для индекса [INDEX]',
                arguments: {
                    INDEX: {type: ArgumentType.NUMBER, defaultValue: 1}
                }
            },
            {
                opcode: 'getConfidence',
                blockType: BlockType.REPORTER,
                text: '📊 Уверенность класса [INDEX] (%)',
                arguments: {
                    INDEX: {type: ArgumentType.NUMBER, defaultValue: 1}
                }
            },
            '---',
            {
                opcode: 'modelInputWidth',
                blockType: BlockType.REPORTER,
                text: 'модель ожидает ширину',
            },
            {
                opcode: 'modelInputHeight',
                blockType: BlockType.REPORTER,
                text: 'модель ожидает высоту',
            },
            {
                opcode: 'isLoaded',
                blockType: BlockType.BOOLEAN,
                text: 'модель готова?',
            }
        ],
        menus: {
            imageSourceMenu: {
                acceptReporters: false,
                items: [
                    { text: 'костюма', value: 'costume' },
                    { text: 'сцены', value: 'backdrop' },
                    { text: 'веб-камеры', value: 'webcam' }
                ]
            }
        }
    };
}

postToWorker(type, payload) {
    const id = this.nextPromiseId++;
    return new Promise((resolve, reject) => {
        this.pendingPromises.set(id, { resolve, reject });
        const transferables = [];
        if (payload && payload.jsonBuffer) transferables.push(payload.jsonBuffer);
        if (payload && payload.weightsBuffer) transferables.push(payload.weightsBuffer);
        this.worker.postMessage({ id, type, payload }, transferables);
    });
}

handleWorkerMessage(event) {
    const { id, type, payload } = event.data;
    if (!this.pendingPromises.has(id)) {
        return;
    }

    const { resolve, reject } = this.pendingPromises.get(id);
    this.pendingPromises.delete(id);

    if (type === 'modelLoaded' || type === 'predictionResult') {
        resolve(payload);
    } else if (type === 'error') {
        reject(new Error(payload));
    }
}

async loadModel() {
    return new Promise(resolve => {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.multiple = true;
        fileInput.accept = '.json,.bin';
        fileInput.click();

        fileInput.onchange = async (e) => {
            const filesList = e.target.files;
            if (!filesList || filesList.length === 0) {
                resolve();
                return;
            }
            
            this.isModelLoaded = false;
            alert('Начинается загрузка модели в фоновом режиме. Интерфейс не зависнет. Откройте консоль (F12) для просмотра процесса.');

            try {
                const allFiles = Array.from(filesList);
                const jsonFile = allFiles.find(f => f.name.toLowerCase().endsWith('.json'));
                const binFiles = allFiles.filter(f => f.name.toLowerCase().endsWith('.bin'));

                if (!jsonFile || binFiles.length === 0) {
                    throw new Error('Необходимо выбрать один .json и сопутствующие .bin файлы.');
                }
                
                const jsonText = await jsonFile.text();
                const modelJson = JSON.parse(jsonText);
                let isModified = false;

                if (modelJson.weightsManifest) {
                    const prefixRegex = /^sequential(_\d+)?\//;
                    modelJson.weightsManifest.forEach(group => {
                        group.weights.forEach(weight => {
                            if (prefixRegex.test(weight.name)) {
                                weight.name = weight.name.replace(prefixRegex, '');
                                isModified = true;
                            }
                        });
                    });
                }

                let layers = null;
                if (modelJson.modelTopology?.model_config?.config?.layers) {
                    layers = modelJson.modelTopology.model_config.config.layers;
                } else if (modelJson.modelTopology?.config?.layers) {
                    layers = modelJson.modelTopology.config.layers;
                }

                if (layers) {
                    for (const layer of layers) {
                        if (layer.class_name === 'InputLayer' && layer.config) {
                            if (layer.config.batch_shape && !layer.config.batch_input_shape) {
                                layer.config.batch_input_shape = layer.config.batch_shape;
                                delete layer.config.batch_shape;
                                isModified = true;
                            }
                        }
                    }
                }
                
                let jsonBuffer;
                if (isModified) {
                    jsonBuffer = new TextEncoder().encode(JSON.stringify(modelJson)).buffer;
                } else {
                    jsonBuffer = await jsonFile.arrayBuffer();
                }
                
                const orderedFileNames = modelJson.weightsManifest.flatMap(group => group.paths);
                
                const binFileMap = new Map(binFiles.map(f => [f.name, f]));

                const orderedBinBuffers = [];
                let totalWeightsSize = 0;
                
                for (const fileName of orderedFileNames) {
                    const file = binFileMap.get(fileName);
                    if (!file) {
                        throw new Error(`Отсутствует необходимый файл с весами: ${fileName}`);
                    }
                    const buffer = await file.arrayBuffer();
                    orderedBinBuffers.push(buffer);
                    totalWeightsSize += buffer.byteLength;
                }
                
                const weightsCombinedBuffer = new Uint8Array(totalWeightsSize);
                let offset = 0;
                for (const buffer of orderedBinBuffers) {
                    weightsCombinedBuffer.set(new Uint8Array(buffer), offset);
                    offset += buffer.byteLength;
                }
                
                const result = await this.postToWorker('loadModel', {
                    jsonBuffer: jsonBuffer,
                    weightsBuffer: weightsCombinedBuffer.buffer
                });
                
                this._modelInputWidth = result.width;
                this._modelInputHeight = result.height;
                this.isModelLoaded = true;

                alert('Модель успешно загружена!');

            } catch (err) {
                this.isModelLoaded = false;
                alert(`Ошибка загрузки: ${err.message}`);
            } finally {
                resolve();
            }
        };
    });
}

loadLabels() {
    return new Promise(resolve => {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.txt';
        fileInput.click();

        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) {
                resolve();
                return;
            }
            try {
                const text = await file.text();
                this.labels = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
                alert('Метки успешно загружены!');
            } catch (err) {
                alert(`Ошибка загрузки меток: ${err.message}`);
            }
            resolve();
        };
    });
}

_getImageDataSource(source) {
    if (source !== 'webcam' && this.runtime.ioDevices.video.videoReady) {
        this.runtime.ioDevices.video.disableVideo();
    }

    switch (source) {
    case 'costume': {
        const target = this.runtime.getEditingTarget();
        if (!target || !target.sprite || !target.sprite.costumes) return null;
        const costume = target.sprite.costumes[target.currentCostume];
        if (!costume) return null;
        
        const skinId = costume.skinId;
        if (skinId && this.runtime.renderer) {
            const skin = this.runtime.renderer._allSkins[skinId];
            if (skin && skin._canvas) {
                return skin._canvas;
            }
        }
        return null;
    }
    case 'backdrop': {
        const stage = this.runtime.getTargetForStage();
        if (!stage || !stage.sprite || !stage.sprite.costumes) return null;
        const backdrop = stage.sprite.costumes[stage.currentCostume];
        if (!backdrop) return null;
        
        const skinId = backdrop.skinId;
        if (skinId && this.runtime.renderer) {
            const skin = this.runtime.renderer._allSkins[skinId];
            if (skin && skin._canvas) {
                return skin._canvas;
            }
        }
        return null;
    }
    case 'webcam': {
        if (!this.runtime.ioDevices.video.videoReady) {
            this.runtime.ioDevices.video.enableVideo();
            return null;
        }
        
        const frame = this.runtime.ioDevices.video.getFrame({
            format: 'canvas',
            dimensions: [this._modelInputWidth, this._modelInputHeight]
        });
        
        return frame;
    }
    default:
        return null;
    }
}

async predictImage(args) {
    if (!this.isModelLoaded) return 0;

    const w = this._modelInputWidth;
    const h = this._modelInputHeight;
    if (w === 0 || h === 0) return 0;

    const imageSource = this._getImageDataSource(Cast.toString(args.IMAGE_SOURCE));
    if (!imageSource) return 0;

    if (this._canvas.width !== w || this._canvas.height !== h) {
        this._canvas.width = w;
        this._canvas.height = h;
    }
    this._ctx.drawImage(imageSource, 0, 0, w, h);
    const imageData = this._ctx.getImageData(0, 0, w, h);

    try {
        const probsData = await this.postToWorker('predict', {
            imageData: imageData.data,
            width: w,
            height: h
        });
        
        this.lastProbabilities = probsData;
        const predictedIndex = probsData.indexOf(Math.max(...this.lastProbabilities));
        return predictedIndex + 1;

    } catch (err) {
        return 0;
    }
}

isLoaded() {
    return this.isModelLoaded;
}

modelInputWidth() {
    return this._modelInputWidth || 0;
}

modelInputHeight() {
    return this._modelInputHeight || 0;
}

getClassName(args) {
    const originalIndex = Cast.toNumber(args.INDEX);
    const index = originalIndex - 1;
    if (this.labels.length === 0) return String(originalIndex);
    if (index >= 0 && index < this.labels.length) {
        return this.labels[index];
    }
    return 'Unknown';
}

getConfidence(args) {
    const index = Cast.toNumber(args.INDEX) - 1;
    if (!this.lastProbabilities || this.lastProbabilities.length === 0) {
        return 0;
    }
    if (index >= 0 && index < this.lastProbabilities.length) {
        const confidence = this.lastProbabilities[index] * 100;
        return parseFloat(confidence.toFixed(2));
    }
    return 0;
}

}

module.exports = Scratch3NeuralNet;