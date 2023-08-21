// Data for the chart
const data = {
  labels: ["High Blood Pressure", "High Cholesterol", "Smoking", "Diabetes", "Obesity", "Physical Inactivity"],
  datasets: [{
    label: "Risk Factors for Heart Attack",
    data: [30, 25, 20, 15, 10, 5], // Replace these values with real data
    backgroundColor: ["#f44336", "#ff9800", "#4caf50", "#2196f3", "#9c27b0", "#00bcd4"],
    borderWidth: 1
  }]
};

// Chart configuration
const config = {
  type: 'bar',
  data: data,
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
};

// Create the chart
window.addEventListener('DOMContentLoaded', (event) => {
  const heartAttackRiskChart = new Chart(document.getElementById("heartAttackRiskChart"), config);
});
