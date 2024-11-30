import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import re
import seaborn as sns
from scipy import stats
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class IPPortPredictor:
    def __init__(self):
        self.console = Console()
        self.seq_length = 10
        self.data = None
        self.port_patterns = {}
        self.port_distribution = None
        self.min_port = 10000  # Updated minimum port
        self.max_port = 63000  # Updated maximum port
        
    def prepare_data(self, json_file):
        """Load and preprocess the JSON data"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.data = df[['port']]
            
            # Analyze port patterns and distribution
            self._analyze_patterns()
            return True
        except Exception as e:
            self.console.print(f"[red]Error preparing data: {str(e)}[/red]")
            return False

    def _analyze_patterns(self):
        """Analyze port patterns and create probability distribution"""
        ports = self.data['port'].values
        
        # Create sequence patterns
        for i in range(len(ports) - self.seq_length):
            seq = tuple(ports[i:i+self.seq_length])
            next_port = ports[i+self.seq_length]
            
            if seq not in self.port_patterns:
                self.port_patterns[seq] = []
            self.port_patterns[seq].append(next_port)
        
        # Create overall port distribution
        self.port_distribution = pd.Series(ports).value_counts().to_dict()
        
        # Calculate port ranges and common differences
        self.port_diffs = pd.Series([ports[i+1] - ports[i] for i in range(len(ports)-1)]).value_counts()
        
        self.console.print("\n[bold cyan]Port Analysis:[/bold cyan]")
        self.console.print(f"Number of unique patterns: {len(self.port_patterns)}")
        self.console.print(f"Number of unique ports: {len(self.port_distribution)}")
        self.console.print(f"Most common port differences: {self.port_diffs.head().to_dict()}")

    def save_model(self, filename):
        """Save the trained model and patterns"""
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Saving model...", total=100)
                
                model_data = {
                    'port_patterns': self.port_patterns,
                    'port_distribution': self.port_distribution,
                    'port_diffs': self.port_diffs
                }
                
                progress.update(task, advance=50)
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
                progress.update(task, advance=50)
                
            self.console.print("[green]Model saved successfully! ✓[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Error saving model: {str(e)}[/red]")
            return False
    
    def load_model(self, filename):
        """Load a previously saved model"""
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Loading model...", total=100)
                
                with open(filename, 'rb') as f:
                    progress.update(task, advance=50)
                    model_data = pickle.load(f)
                    
                self.port_patterns = model_data['port_patterns']
                self.port_distribution = model_data['port_distribution']
                self.port_diffs = model_data['port_diffs']
                progress.update(task, advance=50)
                
            self.console.print("[green]Model loaded successfully! ✓[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Error loading model: {str(e)}[/red]")
            return False

    def predict_next(self, input_sequences, num_predictions=1, stream=False):
        """Predict next ports with optional streaming"""
        if not self.port_patterns:
            self.console.print("[red]No patterns available! Please load and analyze data first.[/red]")
            return None

        predictions = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True if stream else False
        ) as progress:
            task = progress.add_task("Generating predictions...", total=num_predictions)
            
            for i in range(num_predictions):
                prediction = self._generate_single_prediction(input_sequences, i)
                predictions.append(prediction)
                
                if stream:
                    self.console.print(f"Prediction {i+1}: Port {prediction['port']}")
                
                progress.update(task, advance=1)
                if stream:
                    time.sleep(0.01)  # Small delay for streaming effect

        return predictions

    def analyze_predictions(self, predictions):
        """Analyze statistical properties of predictions"""
        pred_ports = np.array([p['port'] for p in predictions])
        
        try:
            mode_result = stats.mode(pred_ports)
            mode_value = mode_result.mode[0] if hasattr(mode_result, 'mode') else mode_result[0]
        except:
            mode_value = "N/A"
        
        stats_dict = {
            'Count': len(pred_ports),
            'Mean': np.mean(pred_ports),
            'Median': np.median(pred_ports),
            'Mode': mode_value,
            'Std Dev': np.std(pred_ports),
            'Min': np.min(pred_ports),
            'Max': np.max(pred_ports),
            'Range': np.ptp(pred_ports),
            '25th Percentile': np.percentile(pred_ports, 25),
            '75th Percentile': np.percentile(pred_ports, 75),
            'IQR': stats.iqr(pred_ports),
            'Skewness': stats.skew(pred_ports),
            'Kurtosis': stats.kurtosis(pred_ports)
        }
        
        return stats_dict

    def plot_evaluation(self, predictions, actual):
        """Plot prediction evaluation results"""
        pred_ports = [p['port'] for p in predictions]
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Time series of predictions
        plt.subplot(2, 1, 1)
        plt.plot(pred_ports, label='Predicted Ports', marker='o', alpha=0.5)
        plt.axhline(y=actual['port'], color='r', linestyle='--', label='Actual Port')
        plt.axhline(y=self.min_port, color='g', linestyle=':', label='Min Port')
        plt.axhline(y=self.max_port, color='g', linestyle=':', label='Max Port')
        plt.title('Port Predictions vs Actual')
        plt.xlabel('Prediction Number')
        plt.ylabel('Port')
        plt.legend()
        
        # Plot 2: Distribution of predictions
        plt.subplot(2, 1, 2)
        sns.histplot(pred_ports, bins=50, kde=True)
        plt.axvline(x=actual['port'], color='r', linestyle='--', label='Actual Port')
        plt.axvline(x=np.mean(pred_ports), color='g', linestyle='--', label='Mean')
        plt.axvline(x=np.median(pred_ports), color='b', linestyle='--', label='Median')
        plt.title('Distribution of Predicted Ports')
        plt.xlabel('Port')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def _generate_single_prediction(self, input_sequences, prediction_number=0):
        """Generate a single port prediction based on input sequence"""
        current_seq = tuple(entry['port'] for entry in input_sequences[-self.seq_length:])
        
        # Try to find the sequence in patterns
        if current_seq in self.port_patterns:
            possible_next = self.port_patterns[current_seq]
            next_port = np.random.choice(possible_next)
        else:
            # If sequence not found, use port distribution and common differences
            last_port = current_seq[-1]
            
            # Get common port differences
            if len(self.port_diffs) > 0:
                common_diffs = self.port_diffs.index.tolist()
                diff_weights = self.port_diffs.values
                diff_weights = diff_weights / diff_weights.sum()
                port_diff = np.random.choice(common_diffs, p=diff_weights)
            else:
                port_diff = np.random.randint(-1000, 1000)
            
            next_port = last_port + port_diff
            
            # Ensure port is within valid range
            next_port = max(self.min_port, min(self.max_port, next_port))
        
        return {
            'port': int(next_port),
            'prediction_number': prediction_number + 1
        }

    def train_model(self, json_file):
        """Train the model on a dataset"""
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Training model...", total=100)
                
                # Load and preprocess data
                progress.update(task, description="[cyan]Loading data...[/cyan]", advance=20)
                success = self.prepare_data(json_file)
                if not success:
                    return False
                
                progress.update(task, description="[cyan]Analyzing patterns...[/cyan]", advance=40)
                # Pattern analysis is done in prepare_data via _analyze_patterns
                
                # Calculate additional statistics
                progress.update(task, description="[cyan]Calculating statistics...[/cyan]", advance=20)
                ports = self.data['port'].values
                self.stats = {
                    'total_samples': len(ports),
                    'unique_ports': len(set(ports)),
                    'unique_patterns': len(self.port_patterns),
                    'min_port_seen': int(min(ports)),
                    'max_port_seen': int(max(ports)),
                    'mean_port': float(np.mean(ports)),
                    'std_port': float(np.std(ports))
                }
                
                progress.update(task, description="[cyan]Finalizing model...[/cyan]", advance=20)
                
            # Print training summary
            self.console.print("\n[bold green]Training Complete! ✓[/bold green]")
            self.console.print("\n[bold cyan]Training Summary:[/bold cyan]")
            self.console.print(f"Total samples processed: {self.stats['total_samples']:,}")
            self.console.print(f"Unique ports observed: {self.stats['unique_ports']:,}")
            self.console.print(f"Unique patterns learned: {self.stats['unique_patterns']:,}")
            self.console.print(f"Port range: {self.stats['min_port_seen']:,} - {self.stats['max_port_seen']:,}")
            self.console.print(f"Mean port: {self.stats['mean_port']:.2f}")
            self.console.print(f"Standard deviation: {self.stats['std_port']:.2f}")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error during training: {str(e)}[/red]")
            return False

def display_menu():
    """Display the main menu"""
    console = Console()
    console.print("\n[bold cyan]=== Port Prediction System ===[/bold cyan]")
    console.print("1. Train New Model")
    console.print("2. Load Existing Model")
    console.print("3. Make Predictions")
    console.print("4. Save Current Model")
    console.print("5. Model Information")
    console.print("6. Exit")
    return input("Select an option (1-6): ")

def parse_ip_port_logs(log_text):
    """Parse ports from log entries"""
    # Clean the input text - remove escape sequences and merge lines if needed
    clean_text = log_text.replace('^[E', '\n').strip()
    
    # Pattern to match IP:PORT
    pattern = r'\d+\.\d+\.\d+\.\d+:(\d+)'
    
    entries = []
    for line in clean_text.split('\n'):
        matches = re.finditer(pattern, line)
        for match in matches:
            entries.append({
                'port': int(match.group(1))
            })
    
    return entries

def get_input_sequences(seq_length):
    """Get input sequences either manually or from pasted logs"""
    print(f"\nEnter the last {seq_length} ports")
    print("Choose input method:")
    print("1. Manual entry")
    print("2. Paste log entries")
    
    choice = input("Select option (1-2): ")
    
    if choice == '1':
        # Manual entry
        input_sequences = []
        for i in range(seq_length):
            port = int(input(f"Enter Port #{i+1}: "))
            input_sequences.append({'port': port})
    else:
        # Paste log entries
        print(f"\nPaste your log entries (at least {seq_length} lines):")
        print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) followed by Enter when done")
        log_lines = []
        try:
            while True:
                line = input()
                log_lines.append(line)
        except EOFError:
            pass
        
        log_text = '\n'.join(log_lines)
        all_entries = parse_ip_port_logs(log_text)
        
        if len(all_entries) < seq_length:
            print(f"Error: Need at least {seq_length} valid entries")
            print(f"Found only {len(all_entries)} valid entries")
            return None
            
        # Take the last seq_length entries
        input_sequences = all_entries[-seq_length:]
        
        print("\nParsed sequences:")
        for i, seq in enumerate(input_sequences, 1):
            print(f"#{i}: Port {seq['port']}")
        
    return input_sequences

def evaluate_prediction(predictions, actual):
    """Evaluate prediction accuracy and find closest match"""
    actual_port = actual['port']
    
    closest_match = None
    min_distance = float('inf')
    exact_match = None
    
    for i, pred in enumerate(predictions):
        pred_port = pred['port']
        distance = abs(actual_port - pred_port)
        
        if distance < min_distance:
            min_distance = distance
            closest_match = (i, pred)
            
        if pred_port == actual_port:
            exact_match = (i, pred)
    
    return exact_match, closest_match

def get_num_predictions():
    """Get number of predictions with error handling and default value"""
    while True:
        try:
            user_input = input("\nEnter number of predictions to make (default 100): ").strip()
            if user_input == '':
                return 100  # default value
            num_pred = int(user_input)
            if num_pred <= 0:
                print("Please enter a positive number")
                continue
            if num_pred > 10000:
                print("Maximum limit is 10000 predictions")
                continue
            return num_pred
        except ValueError:
            print("Please enter a valid number")

def main():
    predictor = IPPortPredictor()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            file_path = input("Enter JSON file path for training: ")
            if predictor.train_model(file_path):
                predictor.console.print("[green]Model trained successfully![/green]")
            else:
                predictor.console.print("[red]Failed to train model![/red]")

        elif choice == '2':
            filename = input("Enter filename to load model (e.g., model.pkl): ")
            predictor.load_model(filename)

        elif choice == '3':
            if predictor.data is None and not predictor.port_patterns:
                predictor.console.print("[red]Please train or load a model first![/red]")
                continue
            
            # Add streaming option
            stream_choice = input("Stream predictions in real-time? (y/n): ").lower()
            stream = stream_choice.startswith('y')
            
            # Get input sequence using new function
            input_sequences = get_input_sequences(predictor.seq_length)
            if input_sequences is None:
                continue
                
            num_pred = get_num_predictions()
            predictions = predictor.predict_next(input_sequences, num_pred, stream=stream)
            
            if predictions:
                # Print first few predictions
                print("\nSample Predictions (first 10):")
                for i, pred in enumerate(predictions[:10], 1):
                    print(f"Prediction {i}: Port {pred['port']}")
                
                # Print statistical analysis
                print("\nStatistical Analysis:")
                stats = predictor.analyze_predictions(predictions)
                for metric, value in stats.items():
                    if isinstance(value, (int, np.integer)):
                        print(f"{metric}: {value:,}")
                    elif isinstance(value, (float, np.floating)):
                        print(f"{metric}: {value:.2f}")
                    else:
                        print(f"{metric}: {value}")
                
                # Get actual result
                while True:
                    try:
                        actual_port = int(input("\nEnter the actual next port: "))
                        if actual_port < predictor.min_port or actual_port > predictor.max_port:
                            print(f"Port must be between {predictor.min_port} and {predictor.max_port}")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid port number")
                
                actual = {'port': actual_port}
                
                # Evaluate predictions
                exact_match, closest_match = evaluate_prediction(predictions, actual)
                
                if exact_match:
                    print(f"\nExact match found! (Prediction #{exact_match[0] + 1})")
                    print(f"Port: {exact_match[1]['port']}")
                else:
                    print(f"\nClosest match was Prediction #{closest_match[0] + 1}:")
                    print(f"Port: {closest_match[1]['port']}")
                    print(f"Difference: {abs(closest_match[1]['port'] - actual['port']):,}")
                    print(f"Percentage Error: {(abs(closest_match[1]['port'] - actual['port']) / actual['port'] * 100):.2f}%")
                
                # Plot evaluation
                predictor.plot_evaluation(predictions, actual)

        elif choice == '4':
            if predictor.port_patterns:
                filename = input("Enter filename to save model (e.g., model.pkl): ")
                predictor.save_model(filename)
            else:
                predictor.console.print("[red]No model to save! Please train or load a model first.[/red]")

        elif choice == '5':
            if hasattr(predictor, 'stats'):
                predictor.console.print("\n[bold cyan]Model Information:[/bold cyan]")
                predictor.console.print(f"Total samples: {predictor.stats['total_samples']:,}")
                predictor.console.print(f"Unique ports: {predictor.stats['unique_ports']:,}")
                predictor.console.print(f"Unique patterns: {predictor.stats['unique_patterns']:,}")
                predictor.console.print(f"Port range: {predictor.stats['min_port_seen']:,} - {predictor.stats['max_port_seen']:,}")
                predictor.console.print(f"Mean port: {predictor.stats['mean_port']:.2f}")
                predictor.console.print(f"Standard deviation: {predictor.stats['std_port']:.2f}")
            else:
                predictor.console.print("[red]No model statistics available. Please train or load a model first.[/red]")

        elif choice == '6':
            predictor.console.print("[yellow]Goodbye![/yellow]")
            break

        else:
            predictor.console.print("[red]Invalid option! Please try again.[/red]")

if __name__ == "__main__":
    main()