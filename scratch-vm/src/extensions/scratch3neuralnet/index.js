const BlockType = require('../../extension-support/block-type');
const ArgumentType = require('../../extension-support/argument-type');
const Cast = require('../../util/cast');
const bindAll = require('lodash.bindall');

// Worker code as a string
const workerCode = `
console.log('[Worker] Начало инициализации...');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
console.log('[Worker] TensorFlow.js загружен, версия:', tf.version.tfjs);

let model = null;
let isBackendReady = false;

(async () => {
    try {
        console.log('[Worker] Попытка инициализации WebGL backend...');
        await tf.setBackend('webgl');
        await tf.ready();
        isBackendReady = true;
        console.log('[Worker] WebGL backend готов');
    } catch (e) {
        console.warn('[Worker] WebGL backend не удалось инициализировать, переход на CPU:', e.message);
        try {
            await tf.setBackend('cpu');
            await tf.ready();
            isBackendReady = true;
            console.log('[Worker] CPU backend готов');
        } catch (e2) {
            console.error('[Worker] Все backends не удалось инициализировать:', e2);
            isBackendReady = false;
        }
    }
    
    if (tf.getBackend() === 'webgl') {
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        console.log('[Worker] WebGL оптимизации применены');
    }
    console.log('[Worker] Текущий backend:', tf.getBackend());
})();

self.onmessage = async (event) => {
    const { id, type, payload } = event.data;
    console.log(\`[Worker] Получено сообщение: \${type}\`);

    try {
        if (type === 'loadModel') {
            const startTime = performance.now();
            console.log('[Worker] Начало загрузки модели...');
            if (!isBackendReady) {
                console.log('[Worker] Backend не готов, ожидание...');
                await tf.ready();
            }
            
            if (model) {
                console.log('[Worker] Очистка предыдущей модели...');
                model.dispose();
                model = null;
            }
            
            const parseStart = performance.now();
            console.log('[Worker] Парсинг JSON модели...');
            const modelJson = JSON.parse(new TextDecoder().decode(payload.jsonBuffer));
            console.log(\`[Worker] JSON модели распарсен за \${(performance.now() - parseStart).toFixed(0)}ms, размер весов: \${payload.weightsBuffer.byteLength} байт\`);
            
            const processStart = performance.now();
            let topology = modelJson.modelTopology;
            
            if (topology) {
                if (topology.model_config && topology.model_config.class_name) {
                    console.log('[Worker] Обнаружена структура model_config');
                    topology = topology.model_config;
                } else if (topology.modelConfig && topology.modelConfig.class_name) {
                    console.log('[Worker] Обнаружена структура modelConfig');
                    topology = topology.modelConfig;
                }

                if (topology.config && topology.config.layers) {
                    console.log('[Worker] Количество слоев:', topology.config.layers.length);
                    let convertedCount = 0;
                    
                    topology.config.layers.forEach((layer, idx) => {
                        console.log(\`[Worker] Обработка слоя \${idx + 1}/\${topology.config.layers.length}: \${layer.class_name}\`);
                        
                        if (layer.config) {
                            delete layer.config.dtype;
                            delete layer.config.kernel_initializer;
                            delete layer.config.bias_initializer;
                            delete layer.config.kernel_regularizer;
                            delete layer.config.bias_regularizer;
                            delete layer.config.activity_regularizer;
                            delete layer.config.kernel_constraint;
                            delete layer.config.bias_constraint;
                        }
                        
                        if (layer.inbound_nodes && Array.isArray(layer.inbound_nodes)) {
                            const needsConversion = layer.inbound_nodes.some(node => 
                                node && typeof node === 'object' && ('args' in node || 'kwargs' in node)
                            );
                            
                            if (needsConversion) {
                                convertedCount++;
                                console.log(\`[Worker]   -> Конвертация inbound_nodes для: \${layer.name}\`);
                                
                                layer.inbound_nodes = layer.inbound_nodes.map((node, nodeIdx) => {
                                    if (!node || typeof node !== 'object') {
                                        console.log(\`[Worker]   -> Node \${nodeIdx}: простой тип, пропуск\`);
                                        return node;
                                    }
                                    
                                    if ('args' in node) {
                                        const args = node.args || [];
                                        
                                        if (Array.isArray(args) && args.length > 0) {
                                            const firstArg = args[0];
                                            
                                            if (firstArg && typeof firstArg === 'object') {
                                                if ('config' in firstArg && 'keras_history' in firstArg.config) {
                                                    const history = firstArg.config.keras_history;
                                                    console.log(\`[Worker]   -> Найден keras_history в config: [\${history.join(', ')}]\`);
                                                    return [[history[0], history[1], history[2]]];
                                                } else if ('keras_history' in firstArg) {
                                                    const history = firstArg.keras_history;
                                                    console.log(\`[Worker]   -> Найден прямой keras_history: [\${history.join(', ')}]\`);
                                                    return [[history[0], history[1], history[2]]];
                                                }
                                            }
                                        }
                                        
                                        console.log(\`[Worker]   -> Неизвестный формат args, используем fallback\`);
                                        return [[]];
                                    }
                                    
                                    console.log(\`[Worker]   -> Node без args, возврат как есть\`);
                                    return node;
                                });
                            }
                        }
                    });
                    
                    if (convertedCount > 0) {
                        console.log(\`[Worker] Конвертировано inbound_nodes для \${convertedCount} слоев\`);
                    }
                }
            }
            console.log(\`[Worker] Обработка топологии завершена за \${(performance.now() - processStart).toFixed(0)}ms\`);

            console.log('[Worker] Создание modelArtifacts...');
            const modelArtifacts = {
                modelTopology: topology,
                weightSpecs: modelJson.weightsManifest.flatMap(group => group.weights),
                weightData: payload.weightsBuffer,
            };
            console.log('[Worker] Количество весовых спецификаций:', modelArtifacts.weightSpecs.length);
            
            let progressTimeout;
            let lastProgressUpdate = 0;
            const progressCallback = (fraction) => {
                const now = performance.now();
                if (now - lastProgressUpdate > 500 || fraction >= 1) {
                    console.log(\`[Worker] Прогресс загрузки: \${(fraction * 100).toFixed(1)}%\`);
                    lastProgressUpdate = now;
                }
                clearTimeout(progressTimeout);
                progressTimeout = setTimeout(() => {
                    console.error('[Worker] Загрузка зависла!');
                    self.postMessage({
                        id: id,
                        type: 'error',
                        payload: 'Ошибка: Загрузка модели зависла (возможно нехватка видеопамяти).'
                    });
                }, 20000);
            };
            
            try {
                const loadStart = performance.now();
                console.log('[Worker] Загрузка модели в TensorFlow.js...');
                
                const ioHandler = {
                    load: async () => {
                        console.log('[Worker] IOHandler.load() вызван');
                        const result = {
                            modelTopology: topology,
                            weightSpecs: modelArtifacts.weightSpecs,
                            weightData: payload.weightsBuffer
                        };
                        console.log('[Worker] IOHandler возвращает данные');
                        return result;
                    }
                };
                
                console.log('[Worker] Вызов tf.loadLayersModel...');
                try {
                    model = await tf.loadLayersModel(ioHandler, { 
                        onProgress: progressCallback,
                        strict: false
                    });
                    console.log('[Worker] Модель загружена успешно!');
                } catch (loadError) {
                    console.error('[Worker] Ошибка при первой попытке загрузки:', loadError.toString());
                    
                    console.log('[Worker] Попытка создания модели вручную через Sequential API...');
                    
                    try {
                        const layers = topology.config.layers;
                        console.log('[Worker] Создание слоев вручную, всего слоев:', layers.length);
                        
                        const snakeToCamel = (str) => str.replace(/_([a-z])/g, (g) => g[1].toUpperCase());
                        
                        const convertConfig = (config) => {
                            const newConfig = {};
                            for (const key in config) {
                                if (config.hasOwnProperty(key)) {
                                    const camelKey = snakeToCamel(key);
                                    let value = config[key];
                                    
                                    if (typeof value === 'string') {
                                        if (value.includes('_')) {
                                            value = snakeToCamel(value);
                                        }
                                        if (key === 'padding' && value === 'valid') {
                                            value = 'valid';
                                        }
                                    }
                                    else if (value && typeof value === 'object' && !Array.isArray(value)) {
                                        value = convertConfig(value);
                                    }
                                    
                                    newConfig[camelKey] = value;
                                }
                            }
                            return newConfig;
                        };
                        
                        const tfLayers = [];
                        
                        for (let i = 1; i < layers.length; i++) {
                            const layer = layers[i];
                            console.log(\`[Worker] Создание слоя \${i}: \${layer.class_name}\`);
                            
                            let config = convertConfig(layer.config);
                            
                            if (i === 1) {
                                config.inputShape = [28, 28, 3];
                            }
                            
                            delete config.name;
                            delete config.trainable;
                            
                            console.log(\`[Worker] Конфиг слоя \${i}:\`, JSON.stringify(config).substring(0, 200));
                            
                            let tfLayer;
                            switch (layer.class_name) {
                                case 'Conv2D':
                                    tfLayer = tf.layers.conv2d(config);
                                    break;
                                case 'MaxPooling2D':
                                    tfLayer = tf.layers.maxPooling2d(config);
                                    break;
                                case 'Flatten':
                                    tfLayer = tf.layers.flatten(config);
                                    break;
                                case 'Dense':
                                    tfLayer = tf.layers.dense(config);
                                    break;
                                default:
                                    console.warn(\`[Worker] Неизвестный тип слоя: \${layer.class_name}\`);
                                    continue;
                            }
                            console.log(\`[Worker] Слой \${i} создан успешно\`);
                            tfLayers.push(tfLayer);
                        }
                        
                        console.log('[Worker] Создание Sequential модели...');
                        model = tf.sequential({ layers: tfLayers });
                        
                        console.log('[Worker] Загрузка весов в модель...');
                        
                        const weightsData = new Float32Array(payload.weightsBuffer);
                        console.log('[Worker] Всего весов (float32):', weightsData.length);
                        
                        const weights = [];
                        let offset = 0;
                        
                        for (let i = 0; i < modelArtifacts.weightSpecs.length; i++) {
                            const spec = modelArtifacts.weightSpecs[i];
                            const size = spec.shape.reduce((a, b) => a * b, 1);
                            const data = weightsData.slice(offset, offset + size);
                            const tensor = tf.tensor(Array.from(data), spec.shape, spec.dtype);
                            weights.push(tensor);
                            console.log(\`[Worker] Вес \${i + 1}/\${modelArtifacts.weightSpecs.length}: \${spec.name}, форма: [\${spec.shape}], размер: \${size}\`);
                            offset += size;
                        }
                        
                        console.log('[Worker] Установка весов в модель...');
                        model.setWeights(weights);
                        
                        console.log('[Worker] Веса загружены, очистка временных тензоров...');
                        weights.forEach(w => w.dispose());
                        
                        console.log('[Worker] Модель создана и веса загружены успешно!');
                    } catch (manualError) {
                        console.error('[Worker] Ошибка при создании модели вручную:', manualError.toString());
                        console.error('[Worker] Stack:', manualError.stack);
                        throw manualError;
                    }
                }
                
                clearTimeout(progressTimeout);
                console.log(\`[Worker] Модель загружена в TF.js за \${(performance.now() - loadStart).toFixed(0)}ms\`);
                
                const inputShape = model.inputs[0].shape;
                console.log('[Worker] Входная форма:', inputShape);
                console.log('[Worker] Выполнение пробного прогона...');
                const dummyInput = tf.zeros([1, inputShape[1] || 28, inputShape[2] || 28, inputShape[3] || 3]);
                const dummyOutput = model.predict(dummyInput);
                console.log('[Worker] Форма выхода:', dummyOutput.shape);
                dummyInput.dispose();
                dummyOutput.dispose();
                console.log('[Worker] Пробный прогон завершен, модель готова');
            } catch (error) {
                clearTimeout(progressTimeout);
                console.error('[Worker] Ошибка при загрузке модели:', error);
                console.error('[Worker] Тип ошибки:', error.constructor.name);
                console.error('[Worker] Сообщение ошибки:', error.message || '(пустое)');
                console.error('[Worker] toString:', error.toString());
                console.error('[Worker] Stack trace:', error.stack);
                
                if (error.message && error.message.includes('Cannot read')) {
                    console.error('[Worker] ДИАГНОСТИКА: Проблема с чтением структуры модели');
                    console.error('[Worker] modelTopology.class_name:', topology?.class_name);
                    console.error('[Worker] modelTopology.config:', topology?.config ? 'существует' : 'отсутствует');
                }
                
                throw new Error('Не удалось загрузить модель: ' + (error.message || error.toString() || 'Несовместимый формат Keras 3'));
            }

            const inputShape = model.inputs[0].shape;
            console.log('[Worker] Входная форма модели:', inputShape);
            console.log(\`[Worker] Общее время загрузки: \${(performance.now() - startTime).toFixed(0)}ms\`);
            
            self.postMessage({ 
                id: id, 
                type: 'modelLoaded', 
                payload: { 
                    height: inputShape[1] || 28,
                    width: inputShape[2] || 28
                } 
            });
            console.log('[Worker] Сообщение modelLoaded отправлено');

        } else if (type === 'predict') {
            const startTime = performance.now();
            console.log('[Worker] Начало предсказания...');
            if (!model) {
                console.error('[Worker] Модель не загружена!');
                throw new Error('Модель не загружена');
            }
            
            const { imageData, width, height } = payload;
            console.log(\`[Worker] Размер изображения: \${width}x\${height}\`);
            console.log(\`[Worker] Тип imageData: \${imageData.constructor.name}\`);
            console.log(\`[Worker] Длина imageData: \${imageData.length}\`);
            console.log(\`[Worker] Первые 20 значений:\`, Array.from(imageData.slice(0, 20)));
            
            if (!imageData || imageData.length === 0) {
                console.error('[Worker] imageData пуст!');
                throw new Error('Пустые данные изображения');
            }
            
            const expectedLength = width * height * 4;
            if (imageData.length !== expectedLength) {
                console.error(\`[Worker] Неверная длина imageData: ожидалось \${expectedLength}, получено \${imageData.length}\`);
                throw new Error(\`Неверный размер данных изображения\`);
            }
            
            // КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Определяем количество каналов из формы модели
            const inputShape = model.inputs[0].shape;
            const expectedChannels = inputShape[3] || 3;
            console.log('[Worker] Входная форма модели:', inputShape);
            console.log('[Worker] Ожидаемое количество каналов:', expectedChannels);
            
            // Статистика входных данных
            let nonZeroCount = 0;
            for (let i = 0; i < imageData.length; i += 4) {
                if (imageData[i] > 0 || imageData[i+1] > 0 || imageData[i+2] > 0) nonZeroCount++;
            }
            console.log(\`[Worker] Не-чёрных пикселей во входных данных: \${nonZeroCount}/\${imageData.length/4}\`);
            
            const tensorStart = performance.now();
            const tensor = tf.tidy(() => {
                const uint8Array = new Uint8Array(imageData);
                console.log('[Worker] Создание тензора из uint8Array...');
                
                // Создаём тензор из RGBA данных
                const imgTensor = tf.tensor3d(uint8Array, [height, width, 4]);
                
                // В зависимости от требуемых каналов, обрабатываем по-разному
                let processedTensor;
                if (expectedChannels === 1) {
                    console.log('[Worker] Конвертация в GRAYSCALE (1 канал)');
                    // Берём только RGB каналы и конвертируем в grayscale
                    const rgb = imgTensor.slice([0, 0, 0], [height, width, 3]);
                    // Формула grayscale: 0.299*R + 0.587*G + 0.114*B
                    const r = rgb.slice([0, 0, 0], [height, width, 1]);
                    const g = rgb.slice([0, 0, 1], [height, width, 1]);
                    const b = rgb.slice([0, 0, 2], [height, width, 1]);
                    processedTensor = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
                } else if (expectedChannels === 3) {
                    console.log('[Worker] Используем RGB (3 канала)');
                    processedTensor = imgTensor.slice([0, 0, 0], [height, width, 3]);
                } else if (expectedChannels === 4) {
                    console.log('[Worker] Используем RGBA (4 канала)');
                    processedTensor = imgTensor;
                } else {
                    console.warn(\`[Worker] Неожиданное количество каналов: \${expectedChannels}, используем RGB\`);
                    processedTensor = imgTensor.slice([0, 0, 0], [height, width, 3]);
                }
                
                const normalized = processedTensor.toFloat().div(255.0);
                const batched = normalized.expandDims(0);
                
                console.log('[Worker] Форма итогового тензора:', batched.shape);
                
                // Логируем статистику тензора
                const tensorData = batched.dataSync();
                const nonZeroTensor = tensorData.filter(v => v > 0.01).length;
                const avgValue = tensorData.reduce((a, b) => a + b, 0) / tensorData.length;
                const maxValue = Math.max(...tensorData);
                const minValue = Math.min(...tensorData);
                console.log(\`[Worker] Статистика тензора:\`);
                console.log(\`[Worker]   Ненулевых значений: \${nonZeroTensor}/\${tensorData.length}\`);
                console.log(\`[Worker]   Среднее: \${avgValue.toFixed(4)}, Мин: \${minValue.toFixed(4)}, Макс: \${maxValue.toFixed(4)}\`);
                
                return batched;
            });
            console.log(\`[Worker] Тензор создан за \${(performance.now() - tensorStart).toFixed(0)}ms, форма: \${tensor.shape}\`);
            
            const predictStart = performance.now();
            console.log('[Worker] Выполнение предсказания...');
            const prediction = model.predict(tensor);
            const probsData = await prediction.data();
            console.log(\`[Worker] Предсказание завершено за \${(performance.now() - predictStart).toFixed(0)}ms, классов: \${probsData.length}\`);
            
            // Детальная информация о предсказании
            const sortedProbs = Array.from(probsData)
                .map((p, i) => ({class: i + 1, prob: p}))
                .sort((a, b) => b.prob - a.prob);
            
            console.log('[Worker] ТОП-3 предсказания:');
            sortedProbs.slice(0, 3).forEach((item, idx) => {
                console.log(\`[Worker]   \${idx + 1}. Класс \${item.class}: \${(item.prob * 100).toFixed(2)}%\`);
            });
            
            tensor.dispose();
            prediction.dispose();
            console.log(\`[Worker] Память освобождена. Общее время: \${(performance.now() - startTime).toFixed(0)}ms\`);
            
            self.postMessage({ 
                id: id, 
                type: 'predictionResult', 
                payload: Array.from(probsData) 
            });
            console.log('[Worker] Результат отправлен');
        }
    } catch (error) {
        console.error('[Worker] Ошибка:', error.message || error.toString());
        console.error('[Worker] Stack:', error.stack);
        
        let errorMessage = 'Неизвестная ошибка воркера';
        if (error.message) {
            errorMessage = error.message;
        } else if (error.toString && error.toString() !== '[object Object]') {
            errorMessage = error.toString();
        }
        
        self.postMessage({
            id: id,
            type: 'error',
            payload: errorMessage
        });
    }
};
`;

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
        
        console.log('[Extension] Инициализация Neural Net Extension...');
        
        try {
            console.log('[Extension] Создание Worker...');
            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(blob);
            this.worker = new Worker(workerUrl);
            
            URL.revokeObjectURL(workerUrl);
            console.log('[Extension] Worker создан успешно');
        } catch (e) {
            console.error('[Extension] Не удалось создать Worker:', e);
            alert('Ошибка: Не удалось создать Worker для нейросети. Проверьте консоль.');
            this.worker = null;
        }

        if (this.worker) {
            this.worker.onmessage = this.handleWorkerMessage;
            this.worker.onerror = (err) => {
                console.error('[Extension] Worker error:', err);
                alert('Ошибка Worker: ' + err.message);
            };
        }
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
        if (!this.worker) {
            console.error('[Extension] Worker не инициализирован');
            return Promise.reject(new Error("Worker not initialized"));
        }
        
        const id = this.nextPromiseId++;
        console.log(`[Extension] Отправка сообщения в Worker: ${type} (id: ${id})`);
        
        return new Promise((resolve, reject) => {
            this.pendingPromises.set(id, { resolve, reject });
            const transferables = [];
            
            if (payload) {
                if (payload.jsonBuffer) {
                    transferables.push(payload.jsonBuffer);
                    console.log(`[Extension] Transferring jsonBuffer: ${payload.jsonBuffer.byteLength} bytes`);
                }
                if (payload.weightsBuffer) {
                    transferables.push(payload.weightsBuffer);
                    console.log(`[Extension] Transferring weightsBuffer: ${payload.weightsBuffer.byteLength} bytes`);
                }
            }
            this.worker.postMessage({ id, type, payload }, transferables);
        });
    }

    handleWorkerMessage(event) {
        const { id, type, payload } = event.data;
        console.log(`[Extension] Получено сообщение от Worker: ${type} (id: ${id})`);
        
        if (!this.pendingPromises.has(id)) {
            console.warn(`[Extension] Promise с id ${id} не найден`);
            return;
        }

        const { resolve, reject } = this.pendingPromises.get(id);
        this.pendingPromises.delete(id);

        if (type === 'modelLoaded' || type === 'predictionResult') {
            console.log(`[Extension] Успешно получен ${type}`);
            resolve(payload);
        } else if (type === 'error') {
            console.error('[Extension] Ошибка от Worker:', payload);
            reject(new Error(payload));
        }
    }

    async loadModel() {
        console.log('[Extension] Запуск loadModel...');
        return new Promise(resolve => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.multiple = true;
            fileInput.accept = '.json,.bin';
            fileInput.click();

            fileInput.onchange = async (e) => {
                const filesList = e.target.files;
                if (!filesList || filesList.length === 0) {
                    console.log('[Extension] Загрузка отменена пользователем');
                    resolve();
                    return;
                }
                
                console.log(`[Extension] Выбрано файлов: ${filesList.length}`);
                this.isModelLoaded = false;
                
                const loadingAlert = document.createElement('div');
                loadingAlert.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:white;padding:20px;border:2px solid #333;border-radius:8px;z-index:10000;box-shadow:0 4px 6px rgba(0,0,0,0.3);font-family:Arial,sans-serif;text-align:center;';
                loadingAlert.innerHTML = '<div style="font-size:18px;font-weight:bold;margin-bottom:10px;">Загрузка модели...</div><div id="progress-text" style="font-size:14px;color:#666;">Чтение файлов...</div>';
                document.body.appendChild(loadingAlert);
                
                const updateProgress = (text) => {
                    const progressText = document.getElementById('progress-text');
                    if (progressText) progressText.textContent = text;
                };
                
                try {
                    const allFiles = Array.from(filesList);
                    const jsonFile = allFiles.find(f => f.name.toLowerCase().endsWith('.json'));
                    const binFiles = allFiles.filter(f => f.name.toLowerCase().endsWith('.bin'));

                    console.log('[Extension] JSON файл:', jsonFile?.name || 'не найден');
                    console.log('[Extension] BIN файлов:', binFiles.length);

                    if (!jsonFile || binFiles.length === 0) {
                        throw new Error('Нужны файлы .json и .bin (выберите их вместе)');
                    }
                    
                    updateProgress('Чтение JSON модели...');
                    console.log('[Extension] Чтение JSON файла...');
                    const jsonText = await jsonFile.text();
                    const modelJson = JSON.parse(jsonText);
                    console.log('[Extension] JSON распарсен');
                    
                    updateProgress('Обработка структуры модели...');
                    let isModified = false;
                    
                    if (modelJson.weightsManifest) {
                        console.log('[Extension] Проверка префиксов весов...');
                        const prefixRegex = /^sequential(_\d+)?\//;
                        modelJson.weightsManifest.forEach(group => {
                            group.weights.forEach(weight => {
                                if (prefixRegex.test(weight.name)) {
                                    weight.name = weight.name.replace(prefixRegex, '');
                                    isModified = true;
                                }
                            });
                        });
                        if (isModified) {
                            console.log('[Extension] Префиксы удалены из имен весов');
                        }
                    }

                    let layers = null;
                    if (modelJson.modelTopology?.model_config?.config?.layers) {
                        layers = modelJson.modelTopology.model_config.config.layers;
                    } else if (modelJson.modelTopology?.config?.layers) {
                        layers = modelJson.modelTopology.config.layers;
                    }

                    if (layers) {
                        console.log('[Extension] Проверка InputLayer...');
                        for (const layer of layers) {
                            if (layer.class_name === 'InputLayer' && layer.config) {
                                if (layer.config.batch_shape && !layer.config.batch_input_shape) {
                                    layer.config.batch_input_shape = layer.config.batch_shape;
                                    delete layer.config.batch_shape;
                                    isModified = true;
                                    console.log('[Extension] batch_shape исправлен на batch_input_shape');
                                }
                            }
                        }
                    }
                    
                    let jsonBuffer;
                    if (isModified) {
                        console.log('[Extension] Сериализация измененного JSON...');
                        jsonBuffer = new TextEncoder().encode(JSON.stringify(modelJson)).buffer;
                    } else {
                        console.log('[Extension] Использование оригинального JSON...');
                        jsonBuffer = await jsonFile.arrayBuffer();
                    }
                    
                    updateProgress(`Чтение весов (${binFiles.length} файлов)...`);
                    console.log('[Extension] Сборка бинарных весов...');
                    const orderedFileNames = modelJson.weightsManifest.flatMap(group => group.paths);
                    const binFileMap = new Map(binFiles.map(f => [f.name, f]));
                    const orderedBinBuffers = [];
                    let totalWeightsSize = 0;
                    
                    for (let i = 0; i < orderedFileNames.length; i++) {
                        const fileName = orderedFileNames[i];
                        updateProgress(`Чтение весов ${i + 1}/${orderedFileNames.length}...`);
                        console.log(`[Extension] Чтение файла весов: ${fileName}`);
                        const file = binFileMap.get(fileName);
                        if (!file) throw new Error(`Не найден файл весов: ${fileName}`);
                        const buffer = await file.arrayBuffer();
                        orderedBinBuffers.push(buffer);
                        totalWeightsSize += buffer.byteLength;
                    }
                    
                    const sizeMB = (totalWeightsSize / 1024 / 1024).toFixed(2);
                    console.log(`[Extension] Общий размер весов: ${sizeMB} MB`);
                    updateProgress(`Объединение весов (${sizeMB} MB)...`);
                    
                    const weightsCombinedBuffer = new Uint8Array(totalWeightsSize);
                    let offset = 0;
                    for (const buffer of orderedBinBuffers) {
                        weightsCombinedBuffer.set(new Uint8Array(buffer), offset);
                        offset += buffer.byteLength;
                    }
                    
                    updateProgress('Загрузка в TensorFlow.js...');
                    console.log('[Extension] Отправка модели в Worker...');
                    const result = await this.postToWorker('loadModel', {
                        jsonBuffer: jsonBuffer,
                        weightsBuffer: weightsCombinedBuffer.buffer
                    });
                    
                    this._modelInputWidth = result.width;
                    this._modelInputHeight = result.height;
                    this.isModelLoaded = true;
                    
                    console.log(`[Extension] Модель загружена! Размер входа: ${result.width}x${result.height}`);
                    
                    document.body.removeChild(loadingAlert);
                    
                    alert(`Модель успешно загружена!\nРазмер входа: ${result.width}x${result.height} пикселей\nРазмер весов: ${sizeMB} MB`);

                } catch (err) {
                    console.error('[Extension] Ошибка загрузки модели:', err);
                    
                    if (loadingAlert && document.body.contains(loadingAlert)) {
                        document.body.removeChild(loadingAlert);
                    }
                    
                    alert('Ошибка загрузки модели: ' + err.message);
                    this.isModelLoaded = false;
                } finally {
                    resolve();
                }
            };
        });
    }

    loadLabels() {
        console.log('[Extension] Запуск loadLabels...');
        return new Promise(resolve => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = '.txt';
            fileInput.click();

            fileInput.onchange = async (e) => {
                const file = e.target.files[0];
                if (!file) {
                    console.log('[Extension] Загрузка меток отменена');
                    resolve();
                    return;
                }
                try {
                    console.log(`[Extension] Чтение файла меток: ${file.name}`);
                    const text = await file.text();
                    this.labels = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
                    console.log(`[Extension] Загружено меток: ${this.labels.length}`);
                    console.log('[Extension] Метки:', this.labels);
                    alert(`Загружено ${this.labels.length} меток классов`);
                } catch (err) {
                    console.error('[Extension] Ошибка загрузки меток:', err);
                    alert('Ошибка загрузки меток: ' + err.message);
                }
                resolve();
            };
        });
    }

    _getImageDataSource(source) {
        console.log(`[Extension] _getImageDataSource вызван для: ${source}`);
        
        if (source !== 'webcam' && this.runtime.ioDevices.video.videoReady) {
            this.runtime.ioDevices.video.disableVideo();
        }

        switch (source) {
        case 'costume': {
            const target = this.runtime.getEditingTarget();
            console.log('[Extension] Получение текущего спрайта...');
            if (!target) {
                console.error('[Extension] Нет активного спрайта');
                return null;
            }
            console.log('[Extension] Спрайт:', target.getName());
            
            if (!target.sprite || !target.sprite.costumes) {
                console.error('[Extension] У спрайта нет костюмов');
                return null;
            }
            
            const costume = target.sprite.costumes[target.currentCostume];
            if (!costume) {
                console.error('[Extension] Костюм не найден');
                return null;
            }
            console.log('[Extension] Текущий костюм:', costume.name);
            
            // Сначала пробуем renderer - это самый надёжный способ
            if (this.runtime.renderer && costume.skinId !== undefined && costume.skinId !== null) {
                console.log('[Extension] skinId:', costume.skinId);
                const skin = this.runtime.renderer._allSkins[costume.skinId];
                if (skin) {
                    console.log('[Extension] Skin найден');
                    
                    // Пробуем разные источники
                    if (skin._canvas) {
                        console.log('[Extension] Используем skin._canvas');
                        return skin._canvas;
                    }
                    if (skin._texture) {
                        if (skin._texture._image) {
                            console.log('[Extension] Используем skin._texture._image');
                            return skin._texture._image;
                        }
                        if (skin._texture.canvas) {
                            console.log('[Extension] Используем skin._texture.canvas');
                            return skin._texture.canvas;
                        }
                    }
                    
                    // Если есть silhouette (это canvas с изображением костюма)
                    if (skin._silhouette && skin._silhouette._canvas) {
                        console.log('[Extension] Используем skin._silhouette._canvas');
                        return skin._silhouette._canvas;
                    }
                }
            }
            
            console.error('[Extension] Не удалось получить изображение костюма');
            console.error('[Extension] СОВЕТ: Убедитесь что костюм отображается на сцене');
            return null;
        }
        case 'backdrop': {
            console.log('[Extension] Получение фона сцены...');
            const stage = this.runtime.getTargetForStage();
            if (!stage || !stage.sprite || !stage.sprite.costumes) {
                console.error('[Extension] Сцена не найдена');
                return null;
            }
            const backdrop = stage.sprite.costumes[stage.currentCostume];
            if (!backdrop) {
                console.error('[Extension] Фон не найден');
                return null;
            }
            console.log('[Extension] Текущий фон:', backdrop.name);
            
            if (this.runtime.renderer && backdrop.skinId !== undefined && backdrop.skinId !== null) {
                console.log('[Extension] skinId:', backdrop.skinId);
                const skin = this.runtime.renderer._allSkins[backdrop.skinId];
                if (skin) {
                    if (skin._canvas) {
                        console.log('[Extension] Используем skin._canvas');
                        return skin._canvas;
                    }
                    if (skin._texture) {
                        if (skin._texture._image) {
                            console.log('[Extension] Используем skin._texture._image');
                            return skin._texture._image;
                        }
                        if (skin._texture.canvas) {
                            console.log('[Extension] Используем skin._texture.canvas');
                            return skin._texture.canvas;
                        }
                    }
                    if (skin._silhouette && skin._silhouette._canvas) {
                        console.log('[Extension] Используем skin._silhouette._canvas');
                        return skin._silhouette._canvas;
                    }
                }
            }
            
            console.error('[Extension] Не удалось получить изображение фона');
            return null;
        }
        case 'webcam': {
            console.log('[Extension] Получение кадра с веб-камеры...');
            if (!this.runtime.ioDevices.video.videoReady) {
                console.log('[Extension] Включаем веб-камеру...');
                this.runtime.ioDevices.video.enableVideo();
                return null;
            }
            const frame = this.runtime.ioDevices.video.getFrame({
                format: 'canvas',
                dimensions: [this._modelInputWidth, this._modelInputHeight]
            });
            console.log('[Extension] Кадр получен:', frame ? 'да' : 'нет');
            return frame;
        }
        default: 
            console.error(`[Extension] Неизвестный источник: ${source}`);
            return null;
        }
    }

    async predictImage(args) {
        console.log('[Extension] Начало распознавания...');
        
        if (!this.isModelLoaded) {
            console.warn('[Extension] Модель не загружена');
            return 0;
        }
        
        const sourceStr = Cast.toString(args.IMAGE_SOURCE);
        console.log(`[Extension] Источник изображения: ${sourceStr}`);
        
        const w = this._modelInputWidth;
        const h = this._modelInputHeight;
        if (w === 0 || h === 0) {
            console.error('[Extension] Размеры модели не определены');
            return 0;
        }

        const imageSource = this._getImageDataSource(sourceStr);
        if (!imageSource) {
            console.warn(`[Extension] Источник изображения "${sourceStr}" недоступен`);
            return 0;
        }
        
        console.log('[Extension] Источник получен, изменение размера...');
        console.log('[Extension] Тип источника:', imageSource.constructor.name);
        console.log('[Extension] Размеры источника:', imageSource.width, 'x', imageSource.height);

        if (this._canvas.width !== w || this._canvas.height !== h) {
            this._canvas.width = w;
            this._canvas.height = h;
        }
        
        try {
            // Очищаем canvas перед рисованием
            this._ctx.clearRect(0, 0, w, h);
            
            // Рисуем с белым фоном для прозрачных изображений
            this._ctx.fillStyle = 'white';
            this._ctx.fillRect(0, 0, w, h);
            
            this._ctx.drawImage(imageSource, 0, 0, w, h);
            
            // ОТЛАДКА: Показываем что рисуем (временно, для проверки)
            console.log('[Extension] Отрисованное изображение:', this._canvas.toDataURL().substring(0, 100) + '...');
            
        } catch (e) {
            console.error('[Extension] Ошибка при отрисовке изображения:', e);
            return 0;
        }
        
        const imageData = this._ctx.getImageData(0, 0, w, h);
        
        console.log(`[Extension] ImageData готов: ${w}x${h}, ${imageData.data.length} байт`);
        
        // Статистика по пикселям
        let sumR = 0, sumG = 0, sumB = 0, countNonZero = 0;
        for (let i = 0; i < imageData.data.length; i += 4) {
            sumR += imageData.data[i];
            sumG += imageData.data[i + 1];
            sumB += imageData.data[i + 2];
            if (imageData.data[i] > 0 || imageData.data[i + 1] > 0 || imageData.data[i + 2] > 0) {
                countNonZero++;
            }
        }
        const totalPixels = imageData.data.length / 4;
        const avgR = (sumR / totalPixels).toFixed(2);
        const avgG = (sumG / totalPixels).toFixed(2);
        const avgB = (sumB / totalPixels).toFixed(2);
        
        console.log('[Extension] Статистика пикселей:');
        console.log(`[Extension]   Средний R: ${avgR}, G: ${avgG}, B: ${avgB}`);
        console.log(`[Extension]   Не-чёрных пикселей: ${countNonZero}/${totalPixels} (${(countNonZero/totalPixels*100).toFixed(1)}%)`);
        console.log('[Extension] Первые 20 значений:', Array.from(imageData.data.slice(0, 20)));
        
        if (countNonZero === 0) {
            console.error('[Extension] ВНИМАНИЕ: Изображение полностью чёрное! Проверьте источник.');
        }
        
        const imageDataCopy = new Uint8ClampedArray(imageData.data);
        console.log('[Extension] Создана копия imageData для передачи в Worker');

        try {
            const probsData = await this.postToWorker('predict', {
                imageData: imageDataCopy,
                width: w,
                height: h
            });
            
            this.lastProbabilities = probsData;
            
            if (!probsData || probsData.length === 0) {
                console.error('[Extension] Получен пустой массив вероятностей');
                return 0;
            }
            
            const predictedIndex = probsData.indexOf(Math.max(...probsData));
            const confidence = (probsData[predictedIndex] * 100).toFixed(2);
            
            console.log('[Extension] ============ РЕЗУЛЬТАТ ============');
            console.log(`[Extension] Предсказанный класс: ${predictedIndex + 1}`);
            console.log(`[Extension] Уверенность: ${confidence}%`);
            console.log('[Extension] Все вероятности:');
            probsData.forEach((p, i) => {
                const percent = (p * 100).toFixed(2);
                const barLength = Math.floor(p * 50);
                const bar = '█'.repeat(barLength);
                console.log(`[Extension]   Класс ${i + 1}: ${percent}% ${bar}`);
            });
            console.log('[Extension] =====================================');
            
            return predictedIndex + 1;

        } catch (err) {
            console.error('[Extension] Ошибка предсказания:', err);
            console.error('[Extension] Stack:', err.stack);
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