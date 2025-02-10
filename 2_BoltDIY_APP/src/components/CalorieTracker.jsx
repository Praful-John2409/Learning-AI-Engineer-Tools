import React, { useState, useEffect } from 'react';

function CalorieTracker() {
  const [foodItem, setFoodItem] = useState('');
  const [calories, setCalories] = useState('');
  const [protein, setProtein] = useState('');
  const [foodLog, setFoodLog] = useState([]);

  useEffect(() => {
    // Load food log from localStorage on component mount
    const storedFoodLog = localStorage.getItem('foodLog');
    if (storedFoodLog) {
      setFoodLog(JSON.parse(storedFoodLog));
    }
  }, []); // Empty dependency array ensures this runs only once on mount

  useEffect(() => {
    // Save food log to localStorage whenever it updates
    localStorage.setItem('foodLog', JSON.stringify(foodLog));
  }, [foodLog]); // This effect runs whenever foodLog changes


  const handleAddFood = () => {
    if (foodItem && calories && protein) {
      const newFoodItem = {
        item: foodItem,
        calories: parseInt(calories),
        protein: parseInt(protein),
      };
      setFoodLog([...foodLog, newFoodItem]);
      setFoodItem('');
      setCalories('');
      setProtein('');
    }
  };

  return (
    <div>
      <h2>Track Calories</h2>
      <div>
        <label htmlFor="food-item">Food Item:</label>
        <input
          type="text"
          id="food-item"
          value={foodItem}
          onChange={(e) => setFoodItem(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="calories">Calories:</label>
        <input
          type="number"
          id="calories"
          value={calories}
          onChange={(e) => setCalories(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="protein">Protein (g):</label>
        <input
          type="number"
          id="protein"
          value={protein}
          onChange={(e) => setProtein(e.target.value)}
        />
      </div>
      <button onClick={handleAddFood}>Add Food</button>

      <h3>Food Log</h3>
      <ul>
        {foodLog.map((item, index) => (
          <li key={index}>
            {item.item} - Calories: {item.calories}, Protein: {item.protein}g
          </li>
        ))}
      </ul>
    </div>
  );
}

export default CalorieTracker;
