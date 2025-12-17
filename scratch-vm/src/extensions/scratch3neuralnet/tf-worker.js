importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

let model = null;
let isBackendReady = false;

(async () => {
    try {
        await tf.setBackend('webgl');
        await tf.ready();
        isBackendReady = true;
    } catch (e) {
        try {
            await tf.setBackend('cpu');
            await tf.ready();
            isBackendReady = true;
        } catch (e2) {
            isBackendReady = false;
        }
    }
    
    if (tf.getBackend() === 'webgl') {
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
    }
})();

self.onmessage = async (event) => {
    const { id, type, payload } = event.data;

    try {
        if (type === 'loadModel') {
            if (!isBackendReady) {
                await tf.ready();
                isBackendReady = true;
            }
            
            if (model) {
                model.dispose();
                model = null;
            }
            
            const modelJson = JSON.parse(new TextDecoder().decode(payload.jsonBuffer));
            
            let topology = modelJson.modelTopology;
            
            if (topology) {
                if (topology.model_config && topology.model_config.class_name) {
                    topology = topology.model_config;
                }
                else if (topology.modelConfig && topology.modelConfig.class_name) {
                    topology = topology.modelConfig;
                }
                
                if (topology.model_config && topology.model_config.class_name) {
                    topology = topology.model_config;
                }

                if (topology.config && topology.config.layers) {
                    topology.config.layers.forEach(layer => {
                        if (layer.inbound_nodes && Array.isArray(layer.inbound_nodes)) {
                            const needsConversion = layer.inbound_nodes.some(node => 
                                node && typeof node === 'object' && ('args' in node || 'kwargs' in node)
                            );
                            
                            if (needsConversion) {
                                layer.inbound_nodes = layer.inbound_nodes.map(node => {
                                    if (!node || typeof node !== 'object') {
                                        return node;
                                    }
                                    
                                    if ('args' in node) {
                                        const args = node.args || [];
                                        const kwargs = node.kwargs || {};
                                        
                                        if (Array.isArray(args)) {
                                            return args.map(arg => {
                                                if (arg && typeof arg === 'object' && 'keras_history' in arg) {
                                                    const history = arg.keras_history;
                                                    return [history[0], history[1], history[2], kwargs];
                                                }
                                                return [arg, 0, 0, kwargs];
                                            });
                                        }
                                    }
                                    
                                    return node;
                                });
                            }
                        }
                    });
                }
            }

            const modelArtifacts = {
                modelTopology: topology,
                weightSpecs: modelJson.weightsManifest.flatMap(group => group.weights),
                weightData: payload.weightsBuffer,
            };
            
            if (!tf.getBackend()) {
                throw new Error('TensorFlow.js backend is not initialized. This should not happen.');
            }
            
            let progressTimeout;

            const progressCallback = (fraction) => {
                clearTimeout(progressTimeout);
                progressTimeout = setTimeout(() => {
                    const errorMessage = `ЗАВИСАНИЕ! Загрузка остановилась более 15 секунд. Вероятная причина: нехватка памяти GPU (WebGL). Попробуйте перезагрузить браузер.`;
                    self.postMessage({
                        id: id,
                        type: 'error',
                        payload: errorMessage
                    });
                }, 15000);
            };
            
            try {
                const ioHandler = tf.io.fromMemory(modelArtifacts);
                model = await tf.loadLayersModel(ioHandler, { onProgress: progressCallback });
                clearTimeout(progressTimeout);
            } catch (error) {
                clearTimeout(progressTimeout);
                throw error;
            }

            const inputShape = model.inputs[0].shape;
            
            const response = {
                height: inputShape[1] || 0,
                width: inputShape[2] || 0
            };
            
            self.postMessage({ id: id, type: 'modelLoaded', payload: response });

        } else if (type === 'predict') {
            if (!model) { 
                throw new Error('Model is not loaded in the worker.'); 
            }
            
            const { imageData, width, height } = payload;
            
            const tensor = tf.tidy(() => {
                const imgTensor = tf.tensor3d(imageData, [height, width, 4])
                                    .slice([0, 0, 0], [height, width, 3]);
                return imgTensor.toFloat().div(255.0).expandDims(0);
            });
            
            const prediction = model.predict(tensor);
            const probsData = await prediction.data();
            
            tensor.dispose();
            prediction.dispose();
            
            self.postMessage({ 
                id: id, 
                type: 'predictionResult', 
                payload: Array.from(probsData) 
            });
        }
    } catch (error) {
        self.postMessage({
            id: id,
            type: 'error',
            payload: error.message
        });
    }
};