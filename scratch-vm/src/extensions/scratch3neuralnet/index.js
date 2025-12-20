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
                        
                        // Удаляем избыточные поля для ускорения
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
                                            
                                            // Проверяем структуру первого аргумента
                                            if (firstArg && typeof firstArg === 'object') {
                                                if ('config' in firstArg && 'keras_history' in firstArg.config) {
                                                    // Формат: {class_name, config: {keras_history: [...]}}
                                                    const history = firstArg.config.keras_history;
                                                    console.log(\`[Worker]   -> Найден keras_history в config: [\${history.join(', ')}]\`);
                                                    return [[history[0], history[1], history[2]]];
                                                } else if ('keras_history' in firstArg) {
                                                    // Формат: {keras_history: [...]}
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
            console.log('[Worker] Структура modelTopology:', JSON.stringify(topology, null, 2).substring(0, 500) + '...');
            
            let progressTimeout;
            let lastProgressUpdate = 0;
            const progressCallback = (fraction) => {
                const now = performance.now();
                // Логируем только каждые 500мс чтобы не спамить
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
                }, 20000); // Увеличено до 20 секунд
            };
            
            try {
                const loadStart = performance.now();
                console.log('[Worker] Загрузка модели в TensorFlow.js...');
                
                // Удаляем training_config - он не нужен для инференса
                const cleanedJson = {
                    format: 'layers-model',
                    generatedBy: 'keras',
                    convertedBy: 'custom',
                    modelTopology: topology,
                    weightsManifest: [{
                        paths: ['weights.bin'],
                        weights: modelArtifacts.weightSpecs
                    }]
                };
                
                console.log('[Worker] Попытка прямой загрузки через tf.loadLayersModel...');
                console.log('[Worker] Проверка топологии перед загрузкой...');
                console.log('[Worker] - class_name:', topology.class_name);
                console.log('[Worker] - backend:', topology.backend);
                console.log('[Worker] - keras_version:', topology.keras_version);
                
                // Попытка 1: Прямая загрузка
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
                        strict: false  // Отключаем строгую проверку
                    });
                    console.log('[Worker] Модель загружена успешно!');
                } catch (loadError) {
                    console.error('[Worker] Ошибка при первой попытке загрузки:', loadError.toString());
                    
                    // Попытка 2: Используем Sequential модель вместо Functional
                    console.log('[Worker] Попытка создания модели вручную через Sequential API...');
                    
                    try {
                        // Создаем модель вручную по слоям
                        const layers = topology.config.layers;
                        console.log('[Worker] Создание слоев вручную, всего слоев:', layers.length);
                        
                        // Функция для конвертации snake_case в camelCase
                        const snakeToCamel = (str) => str.replace(/_([a-z])/g, (g) => g[1].toUpperCase());
                        
                        const convertConfig = (config) => {
                            const newConfig = {};
                            for (const key in config) {
                                if (config.hasOwnProperty(key)) {
                                    const camelKey = snakeToCamel(key);
                                    let value = config[key];
                                    
                                    // Конвертируем специальные строковые значения
                                    if (typeof value === 'string') {
                                        // Конвертируем значения вроде 'channels_last' в 'channelsLast'
                                        if (value.includes('_')) {
                                            value = snakeToCamel(value);
                                        }
                                        // Специальные значения padding
                                        if (key === 'padding' && value === 'valid') {
                                            value = 'valid'; // оставляем как есть
                                        }
                                    }
                                    // Рекурсивно конвертируем вложенные объекты
                                    else if (value && typeof value === 'object' && !Array.isArray(value)) {
                                        value = convertConfig(value);
                                    }
                                    
                                    newConfig[camelKey] = value;
                                }
                            }
                            return newConfig;
                        };
                        
                        const tfLayers = [];
                        
                        for (let i = 1; i < layers.length; i++) { // Пропускаем InputLayer
                            const layer = layers[i];
                            console.log(\`[Worker] Создание слоя \${i}: \${layer.class_name}\`);
                            
                            // Конвертируем конфигурацию из snake_case в camelCase
                            let config = convertConfig(layer.config);
                            
                            if (i === 1) {
                                // Первый слой должен иметь inputShape
                                config.inputShape = [32, 32, 3];
                            }
                            
                            // Удаляем поля которые не нужны для создания слоя
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
                        console.log('[Worker] Распаковка весов из буфера...');
                        
                        // Декодируем веса из ArrayBuffer
                        const weightsData = new Float32Array(payload.weightsBuffer);
                        console.log('[Worker] Всего весов (float32):', weightsData.length);
                        
                        // Разбиваем веса по слоям согласно weightSpecs
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
                
                // Пробный прогон для инициализации
                const inputShape = model.inputs[0].shape;
                console.log('[Worker] Входная форма:', inputShape);
                console.log('[Worker] Выполнение пробного прогона...');
                const dummyInput = tf.zeros([1, inputShape[1] || 32, inputShape[2] || 32, inputShape[3] || 3]);
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
                
                // Пытаемся понять проблему
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
                    height: inputShape[1] || 224,
                    width: inputShape[2] || 224
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
            console.log(\`[Worker] Размер изображения: \${width}x\${height}, пикселей: \${imageData.length / 4}\`);
            
            const tensorStart = performance.now();
            const tensor = tf.tidy(() => {
                const imgTensor = tf.tensor3d(imageData, [height, width, 4])
                                    .slice([0, 0, 0], [height, width, 3]);
                
                return imgTensor.toFloat().div(255.0).expandDims(0);
            });
            console.log(\`[Worker] Тензор создан за \${(performance.now() - tensorStart).toFixed(0)}ms, форма: \${tensor.shape}\`);
            
            const predictStart = performance.now();
            console.log('[Worker] Выполнение предсказания...');
            const prediction = model.predict(tensor);
            const probsData = await prediction.data();
            console.log(\`[Worker] Предсказание завершено за \${(performance.now() - predictStart).toFixed(0)}ms, классов: \${probsData.length}\`);
            
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
        
        // Формируем понятное сообщение об ошибке
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
            // Create worker from inline code
            console.log('[Extension] Создание Worker...');
            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(blob);
            this.worker = new Worker(workerUrl);
            
            // Clean up blob URL after worker is created
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
                if (payload.imageData && payload.imageData.buffer) {
                    transferables.push(payload.imageData.buffer);
                    console.log(`[Extension] Transferring imageData: ${payload.imageData.buffer.byteLength} bytes`);
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
                
                // Показываем начало загрузки
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
                    
                    // Убираем окно загрузки
                    document.body.removeChild(loadingAlert);
                    
                    // Показываем успех
                    alert(`Модель успешно загружена!\nРазмер входа: ${result.width}x${result.height} пикселей\nРазмер весов: ${sizeMB} MB`);

                } catch (err) {
                    console.error('[Extension] Ошибка загрузки модели:', err);
                    
                    // Убираем окно загрузки
                    const loadingAlertToRemove = document.querySelector('div[style*="position:fixed"]');
                    if (loadingAlertToRemove) document.body.removeChild(loadingAlertToRemove);
                    
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
                if (skin && skin._canvas) return skin._canvas;
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
                if (skin && skin._canvas) return skin._canvas;
            }
            return null;
        }
        case 'webcam': {
            if (!this.runtime.ioDevices.video.videoReady) {
                this.runtime.ioDevices.video.enableVideo();
                return null;
            }
            return this.runtime.ioDevices.video.getFrame({
                format: 'canvas',
                dimensions: [this._modelInputWidth, this._modelInputHeight]
            });
        }
        default: return null;
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

        if (this._canvas.width !== w || this._canvas.height !== h) {
            this._canvas.width = w;
            this._canvas.height = h;
        }
        this._ctx.drawImage(imageSource, 0, 0, w, h);
        const imageData = this._ctx.getImageData(0, 0, w, h);
        
        console.log(`[Extension] ImageData готов: ${w}x${h}, ${imageData.data.length} байт`);

        try {
            const probsData = await this.postToWorker('predict', {
                imageData: imageData.data, 
                width: w,
                height: h
            });
            
            this.lastProbabilities = probsData;
            const predictedIndex = probsData.indexOf(Math.max(...probsData));
            const confidence = (probsData[predictedIndex] * 100).toFixed(2);
            
            console.log(`[Extension] Предсказание: класс ${predictedIndex + 1}, уверенность ${confidence}%`);
            console.log('[Extension] Все вероятности:', probsData.map((p, i) => `${i + 1}: ${(p * 100).toFixed(2)}%`).join(', '));
            
            return predictedIndex + 1;

        } catch (err) {
            console.error('[Extension] Ошибка предсказания:', err);
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