import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Box, Button, CircularProgress, Typography, Select, MenuItem, InputLabel, FormControl } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { preprocessData, createDataset, denormalizeQuantity, generateFutureDates } from '../utils/dataPreprocessing';

const SalesPrediction = ({ data }) => {
    const [isTraining, setIsTraining] = useState(false);
    const [predictions, setPredictions] = useState(null);
    const [selectedProduct, setSelectedProduct] = useState('');
    const [error, setError] = useState('');
    const [trainingProgress, setTrainingProgress] = useState({ epoch: 0, loss: 0, totalEpochs: 100 });

    const createModel = () => {
        const model = tf.sequential();
        model.add(tf.layers.lstm({ units: 32, inputShape: [6, 2], returnSequences: false }));
        model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
        model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
        return model;
    };

    const trainModel = async () => {
        setIsTraining(true);
        setError('');

        try {
            const processedData = preprocessData(data);
            const dataset = createDataset(
                processedData.dates,
                processedData.products,
                processedData.quantities
            );

            const model = createModel();

            await model.fit(dataset.inputs, dataset.outputs, {
                epochs: 100,
                batchSize: 32,
                validationSplit: 0.2,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        setTrainingProgress({
                            epoch: epoch + 1,
                            loss: logs.loss.toFixed(4),
                            totalEpochs: 100,
                        });
                    },
                },
            });

            const uniqueProducts = Object.keys(processedData.productEncoder);
            const predictions = uniqueProducts.map((product) => {
                const productId = processedData.productEncoder[product];
                const last6Months = processedData.dates.slice(-6);
                const lastQuantities = processedData.quantities
                    .filter((_, index) => processedData.products[index] === productId)
                    .slice(-6);

                const inputSequence = last6Months.map((date, idx) => [date, lastQuantities[idx]]);
                let currentInput = [...inputSequence];
                const productPredictions = [];

                for (let i = 0; i < 6; i++) {
                    const inputTensor = tf.tensor3d([currentInput], [1, 6, 2]);
                    const prediction = model.predict(inputTensor);
                    const predictedValue = prediction.dataSync()[0];

                    const denormalizedPrediction = denormalizeQuantity(
                        predictedValue,
                        processedData.minQuantity,
                        processedData.maxQuantity
                    );

                    productPredictions.push(denormalizedPrediction);

                    const nextDate = last6Months[last6Months.length - 1] + i + 1;
                    currentInput.shift();
                    currentInput.push([nextDate, denormalizedPrediction]);

                    inputTensor.dispose();
                    prediction.dispose();
                }

                const futureDates = generateFutureDates(
                    new Date(processedData.startDate),
                    6
                );

                return {
                    product,
                    predictions: futureDates.map((date, i) => ({
                        date,
                        quantity: productPredictions[i],
                    })),
                };
            });

            setPredictions(predictions);
        } catch (err) {
            setError('Error training model: ' + err.message);
        } finally {
            setIsTraining(false);
        }
    };

    const getPredictionsForProduct = (product) => {
        if (!product || !predictions) return null;
        return predictions.find((p) => p.product === product);
    };

    const renderChart = (productData) => {
        const chartData = [
            ...data
                .filter((d) => d.product_description === productData.product)
                .map((d) => ({
                    date: d.sales_date,
                    actual: parseInt(d.quantity_sold),
                    predicted: null,
                })),
            ...productData.predictions.map((p) => ({
                date: p.date,
                actual: null,
                predicted: p.quantity,
            })),
        ];

        return (
            <Box key={productData.product} sx={{ mt: 4 }}>
                <Typography variant="h6">{productData.product} - Sales Forecast</Typography>
                <LineChart width={800} height={400} data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ angle: -45 }} height={70} interval={0} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#8884d8"
                        name="Actual Sales"
                        strokeWidth={2}
                    />
                    <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#82ca9d"
                        name="Predicted Sales"
                        strokeDasharray="5 5"
                        strokeWidth={2}
                    />
                </LineChart>
            </Box>
        );
    };

    const uniqueProducts = predictions ? predictions.map((p) => p.product) : [];

    return (
        <Box sx={{ my: 4 }}>
            <Button
                variant="contained"
                onClick={trainModel}
                disabled={isTraining || !data.length}
            >
                {isTraining ? 'Training Model...' : 'Train Model'}
            </Button>

            {isTraining && (
                <Box sx={{ mt: 2 }}>
                    <CircularProgress />
                    <Typography>Training model... Epoch {trainingProgress.epoch}/{trainingProgress.totalEpochs}</Typography>
                    <Typography>Loss: {trainingProgress.loss}</Typography>
                </Box>
            )}

            {error && <Typography color="error">{error}</Typography>}

            <FormControl fullWidth sx={{ mt: 4 }}>
                <InputLabel>Select Product</InputLabel>
                <Select
                    value={selectedProduct}
                    onChange={(e) => setSelectedProduct(e.target.value)}
                    label="Select Product"
                >
                    {uniqueProducts.map((product, index) => (
                        <MenuItem key={index} value={product}>
                            {product}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>

            {selectedProduct && predictions && (
                <Box sx={{ mt: 4 }}>
                    {renderChart(getPredictionsForProduct(selectedProduct))}
                </Box>
            )}
        </Box>
    );
};

export default SalesPrediction;
