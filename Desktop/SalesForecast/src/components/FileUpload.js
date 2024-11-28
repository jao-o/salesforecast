import React, { useState } from 'react';
import Papa from 'papaparse';
import { Button, Box, Alert } from '@mui/material';

const FileUpload = ({ onDataLoaded }) => {
    const [error, setError] = useState('');

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type !== 'text/csv') {
            setError('Please upload a CSV file');
            return;
        }

        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: (results) => {
                if (results.errors.length > 0) {
                    console.error('CSV parsing errors:', results.errors);
                    setError(`Error parsing CSV file: ${results.errors[0].message}`);
                    return;
                }

                const requiredColumns = ['sales_date', 'quantity_sold'];

                // Validate headers
                const headers = Object.keys(results.data[0]).map((header) =>
                    header.trim().toLowerCase()
                );
                const missingColumns = requiredColumns.filter(
                    (col) => !headers.includes(col)
                );

                if (missingColumns.length > 0) {
                    setError(`Missing required columns: ${missingColumns.join(', ')}`);
                    return;
                }

                // Normalize and validate rows
                const validData = results.data
                    .map((row) => {
                        // Normalize headers
                        const normalizedRow = {
                            sales_date: row['sales_date'],
                            quantity_sold: parseFloat(row['quantity_sold']),
                        };

                        // Validate `sales_date`
                        if (
                            !/^\d{4}-\d{2}(-\d{2})?(\s\d{2}:\d{2}:\d{2})?$/.test(
                                normalizedRow.sales_date
                            )
                        ) {
                            console.warn('Invalid sales_date format:', normalizedRow.sales_date);
                            return null;
                        }

                        // Validate `quantity_sold`
                        if (isNaN(normalizedRow.quantity_sold) || normalizedRow.quantity_sold < 0) {
                            console.warn('Invalid quantity_sold value:', normalizedRow.quantity_sold);
                            return null;
                        }

                        return normalizedRow;
                    })
                    .filter((row) => row !== null); // Remove invalid rows

                if (validData.length === 0) {
                    setError('No valid data found in the CSV file');
                    return;
                }

                setError('');
                onDataLoaded(validData);
            },
            error: (error) => {
                console.error('File reading error:', error);
                setError('Error reading file: ' + error.message);
            },
        });
    };

    return (
        <Box sx={{ my: 2 }}>
            <input
                accept=".csv"
                style={{ display: 'none' }}
                id="raised-button-file"
                type="file"
                onChange={handleFileUpload}
            />
            <label htmlFor="raised-button-file">
                <Button variant="contained" component="span">
                    Upload CSV File
                </Button>
            </label>
            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                </Alert>
            )}
        </Box>
    );
};

export default FileUpload;
