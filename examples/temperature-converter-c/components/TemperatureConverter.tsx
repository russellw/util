"use client"

import { useState } from 'react';

const TemperatureConverter = () => {
  const [temperature, setTemperature] = useState('');
  const [fromUnit, setFromUnit] = useState('celsius');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleTemperatureChange = (e) => {
    setTemperature(e.target.value);
    setError('');
  };

  const handleUnitChange = (e) => {
    setFromUnit(e.target.value);
    setError('');
  };

  const convertTemperature = () => {
    if (!temperature.trim()) {
      setError('Please enter a temperature');
      return;
    }

    const tempValue = parseFloat(temperature);
    
    if (isNaN(tempValue)) {
      setError('Please enter a valid number');
      return;
    }

    let convertedTemp;
    let targetUnit;
    
    if (fromUnit === 'celsius') {
      // Convert Celsius to Fahrenheit: (C × 9/5) + 32
      convertedTemp = (tempValue * 9/5) + 32;
      targetUnit = 'Fahrenheit';
    } else {
      // Convert Fahrenheit to Celsius: (F − 32) × 5/9
      convertedTemp = (tempValue - 32) * 5/9;
      targetUnit = 'Celsius';
    }

    setResult({
      original: tempValue,
      converted: convertedTemp.toFixed(2),
      fromUnit,
      targetUnit
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-50 to-blue-100 p-4">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-6">Temperature Converter</h1>
        
        <div className="mb-6">
          <label className="block text-gray-700 mb-2">Temperature</label>
          <input
            type="text"
            value={temperature}
            onChange={handleTemperatureChange}
            className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter temperature"
          />
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
        
        <div className="mb-6">
          <label className="block text-gray-700 mb-2">Convert from</label>
          <select
            value={fromUnit}
            onChange={handleUnitChange}
            className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="celsius">Celsius</option>
            <option value="fahrenheit">Fahrenheit</option>
          </select>
        </div>
        
        <button
          onClick={convertTemperature}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200"
        >
          Convert
        </button>
        
        {result && (
          <div className="mt-8 p-4 bg-blue-50 rounded-lg">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">Result</h2>
            <p className="text-gray-700">
              {result.original}° {result.fromUnit === 'celsius' ? 'Celsius' : 'Fahrenheit'} equals
            </p>
            <p className="text-2xl font-bold text-blue-600">
              {result.converted}° {result.targetUnit}
            </p>
          </div>
        )}
      </div>

      <div className="mt-8 text-sm text-gray-500">
        <p>Formulas used:</p>
        <p>Celsius to Fahrenheit: (C × 9/5) + 32</p>
        <p>Fahrenheit to Celsius: (F − 32) × 5/9</p>
      </div>
    </div>
  );
};

export default TemperatureConverter;