import React from 'react';
import './App.css';
import GoalSetting from './components/GoalSetting';
import CalorieTracker from './components/CalorieTracker';

function App() {
  return (
    <div className="App">
      <h1>Fitness Goal Tracker</h1>
      <div className="component-container">
        <GoalSetting />
      </div>
      <div className="component-container">
        <CalorieTracker />
      </div>
    </div>
  );
}

export default App;
