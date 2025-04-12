// src/components/AgeSlider.jsx - Updated version with arrow controls
import { motion } from 'framer-motion';
import { useRef } from 'react';

const AgeSlider = ({ value, onChange, min = 13, max = 80 }) => {
  const sliderRef = useRef(null);
  
  const handleSliderChange = (e) => {
    onChange(parseInt(e.target.value, 10));
  };

  const decreaseAge = () => {
    if (value > min) {
      onChange(value - 1);
    }
  };

  const increaseAge = () => {
    if (value < max) {
      onChange(value + 1);
    }
  };

  // Calculate the percentage of the slider value
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className="age-slider-container">
      <div className="age-slider-header">
        <span className="age-slider-label">MIN: {min}</span>
        <div className="age-slider-controls">
          <button 
            className="age-arrow-button" 
            onClick={decreaseAge}
            disabled={value <= min}
          >
            «
          </button>
          <span className="age-slider-value">{value} YRS</span>
          <button 
            className="age-arrow-button" 
            onClick={increaseAge}
            disabled={value >= max}
          >
            »
          </button>
        </div>
        <span className="age-slider-label">MAX: {max}</span>
      </div>
      
      <div className="age-slider-track" style={{ position: 'relative' }}>
        <motion.div 
          className="age-slider-progress"
          style={{ width: `${percentage}%` }}
          initial={{ width: '0%' }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.3 }}
        />
        <input
          ref={sliderRef}
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={handleSliderChange}
          className="age-slider-input"
        />
      </div>
      
    </div>
  );
};

export default AgeSlider;