import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def objective_function(x):
    return x**2 - 4*x + 4  # Example: quadratic function

def ant_algorithm(num_ants, num_iterations, lower_bound, upper_bound):
    best_solution = None
    best_value = float('inf')

    for _ in range(num_iterations):
        solutions = np.random.uniform(low=lower_bound, high=upper_bound, size=num_ants)
        values = [objective_function(sol) for sol in solutions]

        # Update pheromones and choose the best solution
        for i in range(num_ants):
            if values[i] < best_value:
                best_solution = solutions[i]
                best_value = values[i]

    return best_solution, best_value

# Particle Swarm Optimization
def particle_swarm_optimization(num_particles, num_iterations, lower_bound, upper_bound):
    # Initialization
    swarm_positions = np.random.uniform(low=lower_bound, high=upper_bound, size=num_particles)
    swarm_velocities = np.random.uniform(low=-1, high=1, size=num_particles)
    personal_best_positions = swarm_positions.copy()
    personal_best_values = [objective_function(pos) for pos in personal_best_positions]
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]

    # PSO main loop
    for _ in range(num_iterations):
        inertia_weight = 0.5
        cognitive_weight = 1.5
        social_weight = 1.5

        # Update velocities and positions
        swarm_velocities = (inertia_weight * swarm_velocities +
                            cognitive_weight * np.random.rand() * (personal_best_positions - swarm_positions) +
                            social_weight * np.random.rand() * (global_best_position - swarm_positions))

        swarm_positions += swarm_velocities

        # Update personal best and global best
        values = [objective_function(pos) for pos in swarm_positions]
        for i in range(num_particles):
            if values[i] < personal_best_values[i]:
                personal_best_positions[i] = swarm_positions[i]
                personal_best_values[i] = values[i]

        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

    return global_best_position, global_best_value

# GUI Application
class OptimizationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.lower_bound = None
        self.upper_bound = None
        self.num_ants = None
        self.num_particles = None
        self.num_iterations = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Optimization Comparison')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.lower_bound_label = QLabel('Lower Bound:')
        self.layout.addWidget(self.lower_bound_label)

        self.lower_bound_input = QLineEdit(self)
        self.layout.addWidget(self.lower_bound_input)

        self.upper_bound_label = QLabel('Upper Bound:')
        self.layout.addWidget(self.upper_bound_label)

        self.upper_bound_input = QLineEdit(self)
        self.layout.addWidget(self.upper_bound_input)

        self.num_ants_label = QLabel('Number of Ants:')
        self.layout.addWidget(self.num_ants_label)

        self.num_ants_input = QLineEdit(self)
        self.layout.addWidget(self.num_ants_input)

        self.num_particles_label = QLabel('Number of Particles:')
        self.layout.addWidget(self.num_particles_label)

        self.num_particles_input = QLineEdit(self)
        self.layout.addWidget(self.num_particles_input)

        self.num_iterations_label = QLabel('Number of Iterations:')
        self.layout.addWidget(self.num_iterations_label)

        self.num_iterations_input = QLineEdit(self)
        self.layout.addWidget(self.num_iterations_input)

        self.load_button = QPushButton('Load Data', self)
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        self.run_button = QPushButton('Run Optimization', self)
        self.run_button.clicked.connect(self.run_optimization)
        self.layout.addWidget(self.run_button)

        self.canvas = MatplotlibCanvas(self, width=5, height=4)
        self.layout.addWidget(self.canvas)

        self.time_label = QLabel('Time: 0 ms')
        self.layout.addWidget(self.time_label)

        self.central_widget.setLayout(self.layout)

    def load_data(self):
        try:
            self.lower_bound = float(self.lower_bound_input.text())
            self.upper_bound = float(self.upper_bound_input.text())
            self.num_ants = int(self.num_ants_input.text())
            self.num_particles = int(self.num_particles_input.text())
            self.num_iterations = int(self.num_iterations_input.text())
        except ValueError:
            print("Invalid input. Please enter valid numeric values.")
            return

    def run_optimization(self):
        if any(param is None for param in [self.lower_bound, self.upper_bound, self.num_ants, self.num_particles, self.num_iterations]):
            print("Please load data first.")
            return

        start_time = time.time()

        # Call Ant Algorithm or PSO with the loaded data
        ant_result = ant_algorithm(num_ants=self.num_ants, num_iterations=self.num_iterations, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        pso_result = particle_swarm_optimization(num_particles=self.num_particles, num_iterations=self.num_iterations, lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        self.time_label.setText(f'Time: {elapsed_time:.2f} ms\nAnt Algorithm Result: {ant_result}\nPSO Result: {pso_result}')


# Matplotlib Canvas for plotting
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    sys.exit(app.exec_())