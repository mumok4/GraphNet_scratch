/* eslint-disable no-mixed-operators */
/* eslint-disable max-len */
const BlockType = require('../../extension-support/block-type');
const ArgumentType = require('../../extension-support/argument-type');
const Cast = require('../../util/cast');
const tf = require('@tensorflow/tfjs');

class Scratch3NeuralNet {
    constructor (runtime) {
        this.runtime = runtime;
        this.model = null;
        this.isModelLoaded = false;
        // Создаем canvas один раз
        this._canvas = document.createElement('canvas');
        this._ctx = this._canvas.getContext('2d', {willReadFrequently: true});
    }

    getInfo () {
        return {
            id: 'scratch3neuralnet',
            name: 'Нейросеть',
            blocks: [
                {
                    opcode: 'loadModel',
                    blockType: BlockType.COMMAND,
                    text: '📁 Загрузить модель (.json и .bin)',
                    arguments: {}
                },
                '---',
                {
                    opcode: 'predictClass',
                    blockType: BlockType.REPORTER,
                    text: '🔮 Предсказать (Вход: [W]x[H])',
                    arguments: {
                        W: {type: ArgumentType.NUMBER, defaultValue: 28},
                        H: {type: ArgumentType.NUMBER, defaultValue: 28}
                    }
                },
                '---',
                {
                    opcode: 'isLoaded',
                    blockType: BlockType.BOOLEAN,
                    text: 'модель готова?',
                    arguments: {}
                }
            ],
            menus: {}
        };
    }

    isLoaded () {
        return this.isModelLoaded;
    }

    loadModel () {
        return new Promise(resolve => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.multiple = true;
            fileInput.accept = '.json,.bin';

            fileInput.onchange = async e => {
                const filesList = e.target.files;
                if (filesList && filesList.length > 0) {
                    try {
                        const filesArray = Array.from(filesList);
                        
                        // --- Patching JSON for Scratch compatibility ---
                        const jsonFileIndex = filesArray.findIndex(f => f.name.toLowerCase().endsWith('.json'));
                        if (jsonFileIndex !== -1) {
                            const jsonFile = filesArray[jsonFileIndex];
                            try {
                                const text = await jsonFile.text();
                                const json = JSON.parse(text);
                                let modified = false;

                                // Fix InputLayer batch_shape
                                if (json.modelTopology?.model_config?.config?.layers) {
                                    for (const layer of json.modelTopology.model_config.config.layers) {
                                        if (layer.class_name === 'InputLayer' && layer.config) {
                                            if (layer.config.batch_shape && !layer.config.batch_input_shape) {
                                                layer.config.batch_input_shape = layer.config.batch_shape;
                                                modified = true;
                                            }
                                        }
                                    }
                                }
                                
                                // Fix Weight names
                                if (json.weightsManifest) {
                                    const prefixRegex = /^sequential(_\d+)?\//;
                                    json.weightsManifest.forEach(group => {
                                        group.weights.forEach(weight => {
                                            if (prefixRegex.test(weight.name)) {
                                                weight.name = weight.name.replace(prefixRegex, '');
                                                modified = true;
                                            }
                                        });
                                    });
                                }

                                if (modified) {
                                    const newContent = JSON.stringify(json);
                                    filesArray[jsonFileIndex] = new File([newContent], jsonFile.name, {type: 'application/json'});
                                }
                            } catch (parseErr) {
                                console.warn('JSON Patch warning:', parseErr);
                            }
                        }
                        // --- End Patching ---

                        this.model = await tf.loadLayersModel(tf.io.browserFiles(filesArray));
                        
                        // Warmup
                        tf.tidy(() => {
                            try {
                                this.model.predict(tf.zeros([1, 28, 28, 1]));
                            } catch (err) {
                                console.log(err);
                            }
                        });

                        this.isModelLoaded = true;
                        console.log('Neural Net: Model Loaded');
                    } catch (err) {
                        console.warn('Neural Net: Load Error', err);
                        this.isModelLoaded = false;
                    }
                }
                resolve();
            };
            fileInput.click();
        });
    }

    _getBoundingBox (ctx, width, height) {
        const imgData = ctx.getImageData(0, 0, width, height);
        const data = imgData.data;
        let minX = width; let minY = height; let maxX = 0; let maxY = 0;
        let found = false;

        // Проходим с шагом 2 для ускорения
        for (let y = 0; y < height; y += 2) {
            for (let x = 0; x < width; x += 2) {
                const alpha = data[(y * width + x) * 4 + 3];
                if (alpha > 20) { // Порог чувствительности
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }

        if (!found) return null;

        // Добавляем небольшой отступ к найденной рамке
        const padding = 2;
        return {
            x: Math.max(0, minX - padding),
            y: Math.max(0, minY - padding),
            w: Math.min(width, maxX - minX + 1 + padding * 2),
            h: Math.min(height, maxY - minY + 1 + padding * 2)
        };
    }

    _debugTensorInConsole (tensorData) {
        let logString = '%c Neural Net Vision: \n';
        const styles = ['color: lime; font-family: monospace; line-height: 10px; font-size: 10px; background: black'];
        
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                const val = tensorData[i * 28 + j];
                // Выбираем символ в зависимости от яркости
                if (val < 0.1) logString += '.';
                else if (val < 0.5) logString += '+';
                else logString += '@';
                logString += ' '; // Пробел для пропорции
            }
            logString += '\n';
        }
        console.log(logString, styles[0]);
    }

    async _getCostumeTensor (targetWidth, targetHeight) {
        const target = this.runtime.getEditingTarget() || this.runtime.getTargetForStage();
        if (!target || !target.sprite) return null;

        const costume = target.sprite.costumes[target.currentCostume];
        const asset = costume.asset;

        const img = new Image();
        const loadPromise = new Promise(resolve => {
            img.onload = resolve;
            img.onerror = () => resolve(false);
        });
        img.src = asset.encodeDataURI();
        await loadPromise;

        // 1. Временный холст для анализа (High Resolution)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = img.width;
        tempCanvas.height = img.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // !!! ТРЮК: Утолщаем линии перед уменьшением !!!
        // Если картинка большая, тени делают линии толще
        tempCtx.shadowColor = 'black';
        tempCtx.shadowBlur = Math.max(5, img.width / 30); // Адаптивная жирность
        tempCtx.drawImage(img, 0, 0);
        // Рисуем еще раз поверх, чтобы ядро линии было четким
        tempCtx.shadowBlur = 0;
        tempCtx.drawImage(img, 0, 0);

        // 2. Ищем границы
        const bbox = this._getBoundingBox(tempCtx, img.width, img.height);

        // 3. Подготовка целевого холста 28x28
        this._canvas.width = targetWidth;
        this._canvas.height = targetHeight;
        this._ctx.fillStyle = 'white'; // Фон белый (чтобы рисовали черным)
        this._ctx.fillRect(0, 0, targetWidth, targetHeight);
        
        // Для качественного сжатия
        this._ctx.imageSmoothingEnabled = true;
        this._ctx.imageSmoothingQuality = 'high';

        if (bbox) {
            // Вписываем в квадрат 20x20 внутри 28x28
            const targetSize = 20;
            const scale = Math.min(targetSize / bbox.w, targetSize / bbox.h);
            
            const drawW = bbox.w * scale;
            const drawH = bbox.h * scale;

            // Центрируем по массе (примерно геометрический центр)
            const offsetX = (targetWidth - drawW) / 2;
            const offsetY = (targetHeight - drawH) / 2;

            this._ctx.drawImage(
                tempCanvas,
                bbox.x, bbox.y, bbox.w, bbox.h, // Source
                offsetX, offsetY, drawW, drawH // Dest
            );
        } else {
            // Пустой холст
            this._ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
        }

        return tf.tidy(() => {
            let tensor = tf.browser.fromPixels(this._canvas);
            tensor = tensor.mean(2); // Grayscale
            tensor = tensor.expandDims(2);
            tensor = tensor.expandDims(0);
            tensor = tensor.div(255.0);
            
            // Инверсия: (Белый фон -> 0, Черная линия -> 1)
            tensor = tf.scalar(1.0).sub(tensor);
            
            // Если включен дебаг, выводим данные
            // (Асинхронно, чтобы не тормозить процесс)
            tensor.data().then(data => this._debugTensorInConsole(data));

            return tensor;
        });
    }

    async predictClass (args) {
        if (!this.isModelLoaded || !this.model) return -1;
        const w = Cast.toNumber(args.W);
        const h = Cast.toNumber(args.H);

        const tensor = await this._getCostumeTensor(w, h);
        if (!tensor) return -1;

        try {
            const prediction = this.model.predict(tensor);
            const indexTensor = prediction.argMax(1);
            const resultIndex = (await indexTensor.data())[0];
            
            // Чистим память
            tensor.dispose();
            prediction.dispose();
            indexTensor.dispose();

            return resultIndex;
        } catch (e) {
            console.error(e);
            if (tensor) tensor.dispose();
            return -1;
        }
    }

    async predictValue (args) {
        if (!this.isModelLoaded || !this.model) return '[]';
        const w = Cast.toNumber(args.W);
        const h = Cast.toNumber(args.H);

        const tensor = await this._getCostumeTensor(w, h);
        if (!tensor) return '[]';

        try {
            const prediction = this.model.predict(tensor);
            const data = await prediction.data();
            const result = Array.from(data);

            tensor.dispose();
            prediction.dispose();

            return JSON.stringify(result.map(x => Number(x.toFixed(2))));
        } catch (e) {
            console.error(e);
            if (tensor) tensor.dispose();
            return 'Error';
        }
    }
}

module.exports = Scratch3NeuralNet;
