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
        
        # Plot 1: Time series of predictions with improved styling
        plt.subplot(2, 1, 1)
        plt.plot(pred_ports, label='Predicted Ports', marker='.', alpha=0.5, 
                 color='skyblue', linestyle='-', markersize=4)
        plt.axhline(y=actual['port'], color='red', linestyle='--', 
                    label='Actual Port', linewidth=2)
        plt.axhline(y=self.min_port, color='green', linestyle=':', 
                    label='Min Port', alpha=0.5)
        plt.axhline(y=self.max_port, color='green', linestyle=':', 
                    label='Max Port', alpha=0.5)
        plt.title('Port Predictions vs Actual', pad=15, fontsize=12)
        plt.xlabel('Prediction Number', fontsize=10)
        plt.ylabel('Port', fontsize=10)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of predictions with improved styling
        plt.subplot(2, 1, 2)
        sns.histplot(pred_ports, bins=30, kde=True, color='skyblue', alpha=0.6)
        plt.axvline(x=actual['port'], color='red', linestyle='--', 
                    label='Actual Port', linewidth=2)
        plt.axvline(x=np.mean(pred_ports), color='green', linestyle='--', 
                    label='Mean', linewidth=2)
        plt.axvline(x=np.median(pred_ports), color='blue', linestyle='--', 
                    label='Median', linewidth=2)
        plt.title('Distribution of Predicted Ports', pad=15, fontsize=12)
        plt.xlabel('Port', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        plt.show()

    def _generate_single_prediction(self, input_sequences, prediction_number=0):
        """Generate a single port prediction based on input sequence"""
        current_seq = tuple(entry['port'] for entry in input_sequences[-self.seq_length:])
        
        # Convert timestamps to naive datetime objects if they aren't already
        timestamps = []
        for entry in input_sequences:
            ts = datetime.fromisoformat(entry['timestamp'])
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            timestamps.append(ts)
        
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                      for i in range(len(timestamps)-1)]
        avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        if current_seq in self.port_patterns:
            possible_next = self.port_patterns[current_seq]
            # Prioritize frequently occurring ports in the pattern
            port_counts = pd.Series(possible_next).value_counts()
            if len(port_counts) > 1:
                # If a port appears significantly more often, prefer it
                if port_counts.iloc[0] > port_counts.iloc[1] * 2:
                    next_port = port_counts.index[0]
                else:
                    next_port = np.random.choice(possible_next)
            else:
                next_port = possible_next[0]
        else:
            last_port = current_seq[-1]
            
            # Look for similar patterns (allowing for small port differences)
            similar_patterns = []
            for pattern in self.port_patterns.keys():
                if all(abs(a - b) <= 10 for a, b in zip(pattern, current_seq)):
                    similar_patterns.extend(self.port_patterns[pattern])
            
            if similar_patterns:
                # Use the most common next port from similar patterns
                next_port = pd.Series(similar_patterns).mode()[0]
            else:
                # Modified port difference calculation
                if len(self.port_diffs) > 0:
                    common_diffs = self.port_diffs.index.tolist()
                    diff_weights = self.port_diffs.values
                    diff_weights = diff_weights / diff_weights.sum()
                    
                    # Add bias correction based on current port position
                    port_range = self.max_port - self.min_port
                    position_in_range = (last_port - self.min_port) / port_range
                    
                    # Adjust weights to prefer negative differences when closer to max_port
                    # and positive differences when closer to min_port
                    adjusted_weights = diff_weights.copy()
                    for i, diff in enumerate(common_diffs):
                        if position_in_range > 0.7 and diff > 0:  # Near max_port
                            adjusted_weights[i] *= 0.5
                        elif position_in_range < 0.3 and diff < 0:  # Near min_port
                            adjusted_weights[i] *= 0.5
                    
                    adjusted_weights = adjusted_weights / adjusted_weights.sum()
                    port_diff = np.random.choice(common_diffs, p=adjusted_weights)
                    
                    # Scale time-based adjustment
                    if avg_time_diff > 60:
                        time_factor = min(1.5, 1 + (avg_time_diff - 60) / 3600)
                        port_diff = int(port_diff * time_factor)
                else:
                    # More controlled random difference when no patterns exist
                    max_diff = min(1000, (self.max_port - self.min_port) // 10)
                    port_diff = np.random.randint(-max_diff, max_diff)
                
                next_port = last_port + port_diff
                
                # Additional boundary adjustment to prevent clustering at edges
                if next_port > self.max_port - 1000:
                    next_port = self.max_port - np.random.randint(1000, 2000)
                elif next_port < self.min_port + 1000:
                    next_port = self.min_port + np.random.randint(1000, 2000)
        
        # Final boundary check
        next_port = max(self.min_port, min(self.max_port, next_port))
        
        return {
            'port': int(next_port),
            'prediction_number': prediction_number + 1,
            'time_factor': avg_time_diff,
            'pattern_match': current_seq in self.port_patterns,
            'similar_pattern_match': bool(similar_patterns) if 'similar_patterns' in locals() else False
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
    console.print("3. Make Continuous Predictions")
    console.print("4. Make Individual Predictions")
    console.print("5. Save Current Model")
    console.print("6. Model Information")
    console.print("7. Exit")
    return input("Select an option (1-7): ")

def parse_ip_port_logs(log_text):
    """Parse IP, port, and timestamp from log entries"""
    # Clean the input text - remove escape sequences and merge lines if needed
    clean_text = log_text.replace('^[E', '\n').strip()
    
    # Pattern to match IP:PORT
    pattern = r'(\d+\.\d+\.\d+\.\d+):(\d+)'
    
    entries = []
    for line in clean_text.split('\n'):
        matches = re.finditer(pattern, line)
        for match in matches:
            entries.append({
                'ip': match.group(1),
                'port': int(match.group(2)),
                'timestamp': datetime.now().isoformat()  # Use current time if not provided
            })
    
    return entries

def get_input_sequences(seq_length):
    """Get input sequences either manually or from pasted logs or JSON"""
    print(f"\nEnter at least {seq_length} records")
    print("Choose input method:")
    print("1. Manual entry")
    print("2. Paste log entries")
    print("3. Load from JSON file")
    
    choice = input("Select option (1-3): ")
    
    if choice == '1':
        # Manual entry
        input_sequences = []
        for i in range(seq_length):
            print(f"\nEntry #{i+1}:")
            ip = input("Enter IP: ")
            port = int(input("Enter Port: "))
            timestamp = datetime.now().isoformat()
            input_sequences.append({
                'ip': ip,
                'port': port,
                'timestamp': timestamp
            })
    
    elif choice == '2':
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
            
        input_sequences = all_entries[-seq_length:]
    
    else:
        # Load from JSON file
        file_path = input("Enter JSON file path: ")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
        
        # Handle both single record and array of records
        if isinstance(data, dict):
            data = [data]
        
        # Convert to list of records with required fields
        all_entries = []
        for record in data:
            entry = {
                'ip': record.get('ip', ''),
                'port': int(record['port']),
                'timestamp': record.get('timestamp', datetime.now().isoformat())
            }
            all_entries.append(entry)
        
        if len(all_entries) < seq_length:
            print(f"Error: Need at least {seq_length} valid entries")
            print(f"Found only {len(all_entries)} valid entries")
            return None
                
        input_sequences = all_entries
    
    print("\nParsed sequences:")
    print(f"Total entries: {len(input_sequences)}")
    print("\nSample of first few entries:")
    for i, seq in enumerate(input_sequences[:5], 1):
        print(f"#{i}: IP: {seq['ip']}, Port: {seq['port']}, Time: {seq['timestamp']}")
    if len(input_sequences) > 5:
        print("...")
    
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

def predict_individual(predictor):
    """Make individual predictions for the next position"""
    if predictor.data is None and not predictor.port_patterns:
        predictor.console.print("[red]Please train or load a model first![/red]")
        return
        
    # Get input sequence
    input_sequences = get_input_sequences(predictor.seq_length)
    if input_sequences is None:
        return
        
    # Get number of predictions to make
    num_pred = get_num_predictions()
    
    # Make all predictions at once
    predictions = predictor.predict_next(input_sequences, num_predictions=num_pred)
    
    # Get actual port
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
            predict_individual(predictor)

        elif choice == '5':
            if predictor.port_patterns:
                filename = input("Enter filename to save model (e.g., model.pkl): ")
                predictor.save_model(filename)
            else:
                predictor.console.print("[red]No model to save! Please train or load a model first.[/red]")

        elif choice == '6':
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

        elif choice == '7':
            predictor.console.print("[yellow]Goodbye![/yellow]")
            break

        else:
            predictor.console.print("[red]Invalid option! Please try again.[/red]")

if __name__ == "__main__":
    main()