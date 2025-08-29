import re
import json
import os
from pathlib import Path

def parse_training_log(log_file_path):
    """
    Parse training log file and extract loss values
    """
    data = []
    
    # Regex pattern to match the log format
    pattern = r'\[Rank \d+\] \(epoch: (\d+), iters: (\d+), time: ([\d.]+), data: ([\d.]+)\) , D_A: ([\d.]+), G_A: ([\d.]+), cycle_A: ([\d.]+), idt_A: ([\d.]+), content_A: ([\d.]+), hue_A: ([\d.]+), D_B: ([\d.]+), G_B: ([\d.]+), cycle_B: ([\d.]+), idt_B: ([\d.]+), content_B: ([\d.]+), hue_B: ([\d.]+)'
    
    with open(log_file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(pattern, line)
            if match:
                epoch, iters, time, data_time, D_A, G_A, cycle_A, idt_A, content_A, hue_A, D_B, G_B, cycle_B, idt_B, content_B, hue_B = match.groups()
                
                data.append({
                    'epoch': int(epoch),
                    'iters': int(iters),
                    'time': float(time),
                    'data': float(data_time),
                    'D_A': float(D_A),
                    'G_A': float(G_A),
                    'cycle_A': float(cycle_A),
                    'idt_A': float(idt_A),
                    'content_A': float(content_A),
                    'hue_A': float(hue_A),
                    'D_B': float(D_B),
                    'G_B': float(G_B),
                    'cycle_B': float(cycle_B),
                    'idt_B': float(idt_B),
                    'content_B': float(content_B),
                    'hue_B': float(hue_B)
                })
            else:
                print(f"Warning: Could not parse line {line_num}: {line}")
    
    print(f"Successfully parsed {len(data)} data points")
    return data

def generate_html_visualization(data, output_file="training_loss_visualization.html"):
    """
    Generate HTML file with interactive loss visualization
    """
    
    # Calculate statistics
    epochs = list(set(d['epoch'] for d in data))
    total_epochs = len(epochs)
    max_iters = max(d['iters'] for d in data)
    avg_time = sum(d['time'] for d in data) / len(data)
    
    # Convert data to JSON for JavaScript
    data_json = json.dumps(data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Loss Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .chart-container {{
            position: relative;
            height: 600px;
            margin: 20px 0;
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .controls {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f8f9fa;
            padding: 12px 18px;
            border-radius: 25px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        .control-group:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}
        
        label {{
            font-weight: 600;
            color: #555;
            font-size: 0.9em;
            cursor: pointer;
            flex: 1;
        }}
        
        input[type="checkbox"] {{
            transform: scale(1.3);
            accent-color: #667eea;
            cursor: pointer;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            transform: perspective(1000px) rotateX(0deg);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: perspective(1000px) rotateX(5deg) translateY(-5px);
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .view-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .view-btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 20px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .view-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .view-btn.active {{
            background: linear-gradient(45deg, #764ba2, #667eea);
        }}
        
        select {{
            padding: 8px 15px;
            border: 2px solid #ddd;
            border-radius: 15px;
            background: white;
            font-size: 0.9em;
            cursor: pointer;
        }}
        
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 10px;
            font-size: 0.85em;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Training Loss Visualization</h1>
        
        <div class="view-controls">
            <button class="view-btn active" onclick="setView('all')">All Losses</button>
            <button class="view-btn" onclick="setView('discriminator')">Discriminator</button>
            <button class="view-btn" onclick="setView('generator')">Generator</button>
            <button class="view-btn" onclick="setView('cycle')">Cycle</button>
            <button class="view-btn" onclick="setView('identity')">Identity</button>
            <button class="view-btn" onclick="setView('content')">Content</button>
            <button class="view-btn" onclick="setView('hue')">Hue</button>
            
            <select id="epochFilter" onchange="filterByEpoch()">
                <option value="all">All Epochs</option>
            </select>
        </div>
        
        <div class="controls">
            <div class="control-group" onclick="toggleCheckbox('showD')">
                <input type="checkbox" id="showD" checked>
                <label for="showD">Discriminator Losses (D_A, D_B)</label>
            </div>
            <div class="control-group" onclick="toggleCheckbox('showG')">
                <input type="checkbox" id="showG" checked>
                <label for="showG">Generator Losses (G_A, G_B)</label>
            </div>
            <div class="control-group" onclick="toggleCheckbox('showCycle')">
                <input type="checkbox" id="showCycle" checked>
                <label for="showCycle">Cycle Losses (cycle_A, cycle_B)</label>
            </div>
            <div class="control-group" onclick="toggleCheckbox('showIdt')">
                <input type="checkbox" id="showIdt" checked>
                <label for="showIdt">Identity Losses (idt_A, idt_B)</label>
            </div>
            <div class="control-group" onclick="toggleCheckbox('showContent')">
                <input type="checkbox" id="showContent" checked>
                <label for="showContent">Content Losses (content_A, content_B)</label>
            </div>
            <div class="control-group" onclick="toggleCheckbox('showHue')">
                <input type="checkbox" id="showHue" checked>
                <label for="showHue">Hue Losses (hue_A, hue_B)</label>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="lossChart"></canvas>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="totalEpochs">{total_epochs}</div>
                <div class="stat-label">Epochs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalIters">{max_iters}</div>
                <div class="stat-label">Max Iterations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgTime">{avg_time:.3f}s</div>
                <div class="stat-label">Avg Time/Iter</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="dataPoints">{len(data)}</div>
                <div class="stat-label">Data Points</div>
            </div>
        </div>
    </div>

    <script>
        const trainingData = {data_json};
        
        const ctx = document.getElementById('lossChart').getContext('2d');
        
        // Color palette for different loss types
        const colors = {{
            D_A: '#FF6B6B', D_B: '#FF8E8E',
            G_A: '#4ECDC4', G_B: '#6EDBDB',
            cycle_A: '#45B7D1', cycle_B: '#6AC7E8',
            idt_A: '#96CEB4', idt_B: '#B8DBC8',
            content_A: '#FFEAA7', content_B: '#FDCB6E',
            hue_A: '#DDA0DD', hue_B: '#E6B8E6'
        }};

        let chart;
        let currentView = 'all';
        let currentEpoch = 'all';

        function createChart(filteredData = trainingData) {{
            if (chart) {{
                chart.destroy();
            }}

            // Create datasets
            const datasets = [];
            const lossTypes = ['D_A', 'D_B', 'G_A', 'G_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'content_A', 'content_B', 'hue_A', 'hue_B'];
            
            lossTypes.forEach(lossType => {{
                datasets.push({{
                    label: lossType,
                    data: filteredData.map(d => ({{x: d.iters, y: d[lossType]}})),
                    borderColor: colors[lossType],
                    backgroundColor: colors[lossType] + '20',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    borderWidth: 2,
                    hidden: false
                }});
            }});

            chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        intersect: false,
                        mode: 'index'
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: `Training Loss Evolution (${{currentEpoch === 'all' ? 'All Epochs' : 'Epoch ' + currentEpoch}})`,
                            font: {{
                                size: 20,
                                weight: 'bold'
                            }},
                            color: '#2c3e50'
                        }},
                        legend: {{
                            display: true,
                            position: 'top',
                            labels: {{
                                usePointStyle: true,
                                padding: 12,
                                font: {{
                                    size: 10
                                }}
                            }}
                        }},
                        tooltip: {{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: 'white',
                            bodyColor: 'white',
                            borderColor: '#667eea',
                            borderWidth: 1,
                            cornerRadius: 8,
                            displayColors: true,
                            callbacks: {{
                                title: function(context) {{
                                    const dataPoint = filteredData[context[0].dataIndex];
                                    return `Epoch ${{dataPoint.epoch}}, Iter ${{dataPoint.iters}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'linear',
                            title: {{
                                display: true,
                                text: 'Iterations',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.1)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Loss Value',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.1)'
                            }},
                            beginAtZero: true
                        }}
                    }},
                    animation: {{
                        duration: 1000,
                        easing: 'easeInOutCubic'
                    }}
                }}
            }});
            
            // Apply current view
            applyView(currentView);
        }}

        function toggleCheckbox(id) {{
            const checkbox = document.getElementById(id);
            checkbox.checked = !checkbox.checked;
            checkbox.dispatchEvent(new Event('change'));
        }}

        function toggleLossGroup(groupName, show) {{
            const patterns = {{
                'D': ['D_A', 'D_B'],
                'G': ['G_A', 'G_B'],
                'cycle': ['cycle_A', 'cycle_B'],
                'idt': ['idt_A', 'idt_B'],
                'content': ['content_A', 'content_B'],
                'hue': ['hue_A', 'hue_B']
            }};
            
            const lossesToToggle = patterns[groupName];
            chart.data.datasets.forEach((dataset, index) => {{
                if (lossesToToggle.includes(dataset.label)) {{
                    const meta = chart.getDatasetMeta(index);
                    meta.hidden = !show;
                }}
            }});
            chart.update('none');
        }}

        function setView(viewType) {{
            currentView = viewType;
            
            // Update button states
            document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            applyView(viewType);
        }}

        function applyView(viewType) {{
            const allLosses = ['D_A', 'D_B', 'G_A', 'G_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'content_A', 'content_B', 'hue_A', 'hue_B'];
            const views = {{
                'all': allLosses,
                'discriminator': ['D_A', 'D_B'],
                'generator': ['G_A', 'G_B'],
                'cycle': ['cycle_A', 'cycle_B'],
                'identity': ['idt_A', 'idt_B'],
                'content': ['content_A', 'content_B'],
                'hue': ['hue_A', 'hue_B']
            }};
            
            const visibleLosses = views[viewType] || allLosses;
            
            chart.data.datasets.forEach((dataset, index) => {{
                const meta = chart.getDatasetMeta(index);
                meta.hidden = !visibleLosses.includes(dataset.label);
            }});
            
            chart.update('none');
        }}

        function filterByEpoch() {{
            const epochFilter = document.getElementById('epochFilter').value;
            currentEpoch = epochFilter;
            
            let filteredData = trainingData;
            if (epochFilter !== 'all') {{
                filteredData = trainingData.filter(d => d.epoch === parseInt(epochFilter));
            }}
            
            createChart(filteredData);
        }}

        function populateEpochFilter() {{
            const epochs = [...new Set(trainingData.map(d => d.epoch))].sort((a, b) => a - b);
            const select = document.getElementById('epochFilter');
            
            epochs.forEach(epoch => {{
                const option = document.createElement('option');
                option.value = epoch;
                option.textContent = `Epoch ${{epoch}}`;
                select.appendChild(option);
            }});
        }}

        // Event listeners
        document.getElementById('showD').addEventListener('change', (e) => {{
            toggleLossGroup('D', e.target.checked);
        }});
        
        document.getElementById('showG').addEventListener('change', (e) => {{
            toggleLossGroup('G', e.target.checked);
        }});
        
        document.getElementById('showCycle').addEventListener('change', (e) => {{
            toggleLossGroup('cycle', e.target.checked);
        }});
        
        document.getElementById('showIdt').addEventListener('change', (e) => {{
            toggleLossGroup('idt', e.target.checked);
        }});
        
        document.getElementById('showContent').addEventListener('change', (e) => {{
            toggleLossGroup('content', e.target.checked);
        }});
        
        document.getElementById('showHue').addEventListener('change', (e) => {{
            toggleLossGroup('hue', e.target.checked);
        }});

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            populateEpochFilter();
            createChart();
        }});
    </script>
</body>
</html>"""

    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML visualization saved to: {output_file}")
    return output_file

def main():
    """
    Main function to process training log and generate visualization
    """
    log_file = "/home/as76usr/sigtuple/Eshwar/Garuda-model-dev/AS76/ISP/Defocus-Deblur/pytorch-CycleGAN-and-pix2pix/checkpoints/resnet_2_blocks/loss_log.txt"
    
    # Check if file exists
    if not os.path.exists(log_file):
        print(f"Error: File '{log_file}' not found!")
        print("Please make sure the file exists and try again.")
        return
    
    try:
        # Parse the log file
        print(f"Parsing training log: {log_file}")
        data = parse_training_log(log_file)
        
        if not data:
            print("No data found in the log file!")
            return
        
        output_file = os.path.join(os.path.dirname(log_file),"training_loss_visualization.html" )
        
        generate_html_visualization(data, output_file)
        
        print(f"\\n✅ Success!")
        print(f"📊 Processed {len(data)} data points")
        print(f"📁 HTML file saved: {output_file}")
        print(f"🌐 Open the HTML file in your browser to view the interactive plot")
        
        # Print some basic statistics
        epochs = sorted(list(set(d['epoch'] for d in data)))
        print(f"\\n📈 Training Summary:")
        print(f"   Epochs: {min(epochs)} - {max(epochs)} ({len(epochs)} total)")
        print(f"   Iterations: {min(d['iters'] for d in data)} - {max(d['iters'] for d in data)}")
        print(f"   Average time per iteration: {sum(d['time'] for d in data) / len(data):.3f}s")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Please check your log file format and try again.")

if __name__ == "__main__":
    main()