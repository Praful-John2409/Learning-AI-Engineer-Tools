import React, { useState } from 'react';

function GoalSetting() {
  const [goalType, setGoalType] = useState('');
  const [targetWeight, setTargetWeight] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (goalType && targetWeight) {
      const goalData = { goalType, targetWeight };
      localStorage.setItem('fitnessGoal', JSON.stringify(goalData));
      setSuccessMessage('Goal saved successfully!');
      setTimeout(() => setSuccessMessage(''), 3000); // Clear message after 3 seconds
    } else {
      alert('Please select a goal type and enter a target weight.');
    }
  };

  return (
    <div>
      <h2>Set Your Goals</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="goal-type">Goal Type:</label>
          <select
            id="goal-type"
            value={goalType}
            onChange={(e) => setGoalType(e.target.value)}
          >
            <option value="">Select Goal</option>
            <option value="weight-loss">Weight Loss</option>
            <option value="muscle-gain">Muscle Gain</option>
            <option value="general-fitness">General Fitness</option>
          </select>
        </div>
        <div>
          <label htmlFor="target-weight">Target Weight (kg):</label>
          <input
            type="number"
            id="target-weight"
            value={targetWeight}
            onChange={(e) => setTargetWeight(e.target.value)}
          />
        </div>
        <button type="submit">Set Goal</button>
        {successMessage && <p className="success-message">{successMessage}</p>}
      </form>
    </div>
  );
}

export default GoalSetting;
