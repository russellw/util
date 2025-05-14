// components/TemperatureConverter.tsx
'use client'; // This directive is necessary for components with interactivity

import React, { useState, ChangeEvent } from 'react';

const TemperatureConverter: React.FC = () => {
  const [celsius, setCelsius] = useState<string>('');
  const [fahrenheit, setFahrenheit] = useState<string>('');

  const handleCelsiusChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setCelsius(value);
    if (value === '') {
      setFahrenheit('');
      return;
    }
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setFahrenheit(((numValue * 9) / 5 + 32).toFixed(2));
    } else {
      setFahrenheit('Invalid Input');
    }
  };

  const handleFahrenheitChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setFahrenheit(value);
    if (value === '') {
      setCelsius('');
      return;
    }
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setCelsius((((numValue - 32) * 5) / 9).toFixed(2));
    } else {
      setCelsius('Invalid Input');
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h1 className="text-2xl font-bold text-center text-blue-600">Temperature Converter</h1>
      <div>
        <label htmlFor="celsius" className="block text-sm font-medium text-gray-700">
          Celsius (°C)
        </label>
        <input
          type="number"
          id="celsius"
          value={celsius}
          onChange={handleCelsiusChange}
          placeholder="Enter Celsius"
          className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
        />
      </div>
      <div>
        <label htmlFor="fahrenheit" className="block text-sm font-medium text-gray-700">
          Fahrenheit (°F)
        </label>
        <input
          type="number"
          id="fahrenheit"
          value={fahrenheit}
          onChange={handleFahrenheitChange}
          placeholder="Enter Fahrenheit"
          className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
        />
      </div>
      {(celsius === 'Invalid Input' || fahrenheit === 'Invalid Input') && (
         <p className="text-red-500 text-sm text-center">Please enter a valid number.</p>
      )}
    </div>
  );
};

export default TemperatureConverter;