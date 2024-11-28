import * as tf from '@tensorflow/tfjs';

// Convert sales_date (YYYY-MM-DD HH:mm:ss) into numerical format (starting from 1 for 2024-01)
export const preprocessData = (data) => {
    // Extract dates and convert to numerical format (using only year and month)
    const dates = data.map(row => row.sales_date.split(' ')[0]); // Extract the date part (YYYY-MM-DD)
    
    const startDate = new Date(Math.min(...dates.map(d => new Date(d)))); // Find the earliest date (ignoring time)
    
    const processedDates = dates.map(date => {
        const currentDate = new Date(date);  // Parse the date
        const monthsDiff = (currentDate.getFullYear() - startDate.getFullYear()) * 12 
            + (currentDate.getMonth() - startDate.getMonth());
        return monthsDiff + 1; // We start numbering from 1 for the first month
    });

    // Handle both numeric and string product descriptions
    const uniqueProducts = [...new Set(data.map(row => row.product_description))];
    const productEncoder = {};

    // Check if product_description is numeric or string and encode accordingly
    uniqueProducts.forEach((product, index) => {
        productEncoder[product] = typeof product === 'number' ? product : index; // Keep numeric IDs as they are
    });

    const encodedProducts = data.map(row => productEncoder[row.product_description]);

    // Normalize quantities (sales) between 0 and 1
    const quantities = data.map(row => parseFloat(row.quantity_sold));
    const minQuantity = Math.min(...quantities);
    const maxQuantity = Math.max(...quantities);
    const normalizedQuantities = quantities.map(q => 
        (q - minQuantity) / (maxQuantity - minQuantity)
    );

    return {
        dates: processedDates,
        products: encodedProducts,
        quantities: normalizedQuantities,
        productEncoder,
        minQuantity,
        maxQuantity,
        startDate
    };
};

// Create the dataset for training (X, y)
export const createDataset = (dates, products, quantities, windowSize = 6) => {
    const X = [];
    const y = [];

    // Sliding window approach to create input/output pairs
    for (let i = 0; i < dates.length - windowSize; i++) {
        const window = [];
        for (let j = 0; j < windowSize; j++) {
            window.push([dates[i + j], products[i + j]]);
        }
        X.push(window);
        y.push(quantities[i + windowSize]); // The target is the quantity sold in the next month
    }

    return {
        inputs: tf.tensor3d(X),  // Shape: [num_samples, window_size, 2]
        outputs: tf.tensor2d(y, [y.length, 1])  // Shape: [num_samples, 1]
    };
};

// Denormalize the predicted quantity to its original scale
export const denormalizeQuantity = (normalizedValue, minQuantity, maxQuantity) => {
    return Math.round(normalizedValue * (maxQuantity - minQuantity) + minQuantity);
};

// Generate future dates for the next `numMonths` months after a given `startDate`
export const generateFutureDates = (startDate, numMonths) => {
    const dates = [];
    const currentDate = new Date(startDate);
    
    for (let i = 1; i <= numMonths; i++) {
        currentDate.setMonth(currentDate.getMonth() + 1);
        const year = currentDate.getFullYear();
        const month = String(currentDate.getMonth() + 1).padStart(2, '0');
        dates.push(`${year}-${month}`);
    }
    
    return dates;
};

// Function to predict sales for the next 6 months for a specific product
export const predictSales = async (model, productDescription, data, windowSize = 6) => {
    // Filter data for the selected product
    const selectedProduct = data.filter(row => row.product_description === productDescription);
    const { dates, products, quantities, productEncoder, minQuantity, maxQuantity, startDate } = preprocessData(selectedProduct);
    
    // Convert the last 6 months data into input format for prediction
    const last6Months = dates.slice(-windowSize);
    const last6Products = products.slice(-windowSize);
    const input = last6Months.map((date, idx) => [date, last6Products[idx]]);
    
    // Prepare input tensor
    const inputTensor = tf.tensor3d([input], [1, windowSize, 2]);

    // Predict the next quantity using the model
    const predictedNormalized = model.predict(inputTensor).dataSync()[0];

    // Denormalize the predicted quantity
    const predictedQuantity = denormalizeQuantity(predictedNormalized, minQuantity, maxQuantity);
    
    // Generate future dates for the next 6 months
    const futureDates = generateFutureDates(startDate, 6);
    
    return { predictedQuantity, futureDates };
};
