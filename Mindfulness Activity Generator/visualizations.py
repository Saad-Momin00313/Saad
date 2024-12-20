import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

def create_mood_chart(mood_data):
    """Create an interactive mood tracking chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=mood_data['dates'],
        y=mood_data['mood'],
        name='Mood',
        line=dict(color='#FF69B4', width=2),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=mood_data['dates'],
        y=mood_data['energy'],
        name='Energy',
        line=dict(color='#4CAF50', width=2),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Mood & Energy Tracking',
        xaxis_title='Date',
        yaxis_title='Level',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_practice_heatmap(practice_data):
    """Create a practice heatmap calendar."""
    fig = go.Figure(data=go.Heatmap(
        z=practice_data['values'],
        x=practice_data['dates'],
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Practice Intensity Calendar',
        height=300
    )
    
    return fig

def create_progress_chart(progress_data):
    """Create a progress tracking chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=progress_data['dates'],
        y=progress_data['minutes'],
        name='Practice Minutes',
        marker_color='#64B5F6'
    ))
    
    fig.add_trace(go.Scatter(
        x=progress_data['dates'],
        y=progress_data['goal_line'],
        name='Daily Goal',
        line=dict(color='#FFB74D', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Daily Practice Progress',
        xaxis_title='Date',
        yaxis_title='Minutes',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_activity_flow_diagram(activity_data):
    """Create an interactive activity flow diagram."""
    nodes = [
        "Start",
        "Breathing",
        "Movement",
        "Meditation",
        "Reflection",
        "Complete"
    ]
    
    # Node positions in a circular layout
    node_positions = {
        node: (
            0.5 + 0.4 * np.cos(2 * np.pi * i / (len(nodes)-1)),
            0.5 + 0.4 * np.sin(2 * np.pi * i / (len(nodes)-1))
        )
        for i, node in enumerate(nodes)
    }
    
    # Create edges
    edge_x = []
    edge_y = []
    for i in range(len(nodes)-1):
        x0, y0 = node_positions[nodes[i]]
        x1, y1 = node_positions[nodes[i+1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    node_x = [pos[0] for pos in node_positions.values()]
    node_y = [pos[1] for pos in node_positions.values()]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC'],
            line=dict(width=2, color='white')
        ),
        text=nodes,
        textposition="middle center",
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        title='Mindfulness Practice Flow',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def create_sequence_diagram(activity):
    """Create a sequence diagram for the activity steps."""
    steps = [step.strip() for step in activity['text'].split('STEPS:')[1].split('BENEFITS:')[0].split('\n') if step.strip()]
    
    fig = go.Figure()
    
    # Add steps as boxes
    for i, step in enumerate(steps):
        if not step.startswith(str(i+1)):
            continue
            
        fig.add_trace(go.Scatter(
            x=[0.2, 0.8],
            y=[len(steps)-i, len(steps)-i],
            mode='lines',
            line=dict(width=20, color=f'hsl({360*i/len(steps)}, 70%, 50%)'),
            hoverinfo='text',
            hovertext=step,
            name=f'Step {i+1}'
        ))
        
        # Add step text
        fig.add_annotation(
            x=0.5,
            y=len(steps)-i,
            text=f'Step {i+1}',
            showarrow=False,
            font=dict(color='white')
        )
    
    # Update layout
    fig.update_layout(
        title='Practice Sequence',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, len(steps)+1]),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def create_practice_guide(activity):
    """Create an interactive practice guide with proper HTML rendering."""
    steps = [step.strip() for step in activity['text'].split('STEPS:')[1].split('BENEFITS:')[0].split('\n') if step.strip()]
    
    # Create step elements
    step_elements = []
    for step in steps:
        if step.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
            step_elements.append(
                '<div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; '
                'margin-bottom: 10px;">' + step + '</div>'
            )
    
    # Create the HTML template with embedded JavaScript
    guide_html = f'''
    <div style="background: linear-gradient(145deg, #1a1a1a, #2a2a2a); border-radius: 15px; padding: 20px; color: white; margin: 20px 0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: white;">{activity['category']} Practice Guide</h2>
            <p style="color: #64B5F6; font-size: 1.2em;">Duration: {activity.get('duration', 10)} minutes</p>
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 15px;">
            {''.join(step_elements)}
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <div id="timer-display" style="font-size: 2em; color: #64B5F6; margin-bottom: 10px;">
                {str(activity.get('duration', 10)).zfill(2)}:00
            </div>
            <div id="timer-circle" style="width: 60px; height: 60px; border: 3px solid #64B5F6; 
                border-radius: 50%; margin: 20px auto; transition: all 0.3s ease;">
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const duration = {activity.get('duration', 10)};
            let timeLeft = duration * 60;
            let timerInterval = null;
            let isRunning = false;
            
            const timerDisplay = document.getElementById('timer-display');
            const timerCircle = document.getElementById('timer-circle');
            
            function updateTimerDisplay() {{
                const minutes = Math.floor(timeLeft / 60);
                const seconds = timeLeft % 60;
                timerDisplay.textContent = 
                    String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
            }}
            
            function startTimer() {{
                if (!isRunning) {{
                    isRunning = true;
                    timerCircle.style.animation = 'pulse 2s infinite';
                    timerCircle.style.borderColor = '#64B5F6';
                    
                    timerInterval = setInterval(() => {{
                        if (timeLeft > 0) {{
                            timeLeft--;
                            updateTimerDisplay();
                        }} else {{
                            clearInterval(timerInterval);
                            isRunning = false;
                            timerDisplay.style.color = '#4CAF50';
                            timerDisplay.textContent = 'Complete!';
                            timerCircle.style.animation = 'none';
                            timerCircle.style.borderColor = '#4CAF50';
                        }}
                    }}, 1000);
                }}
            }}
            
            function pauseTimer() {{
                isRunning = false;
                clearInterval(timerInterval);
                timerCircle.style.animation = 'none';
            }}
            
            function resetTimer() {{
                isRunning = false;
                clearInterval(timerInterval);
                timeLeft = duration * 60;
                timerDisplay.style.color = '#64B5F6';
                timerCircle.style.animation = 'none';
                timerCircle.style.borderColor = '#64B5F6';
                updateTimerDisplay();
            }}
            
            // Expose functions to window object
            window.startTimer = startTimer;
            window.pauseTimer = pauseTimer;
            window.resetTimer = resetTimer;
        }});
    </script>
    
    <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.2); opacity: 0.7; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
    </style>
    '''
    
    return guide_html

def create_body_scan_visualization(activity):
    """Create an interactive body scan visualization."""
    body_parts = [
        'Head', 'Neck', 'Shoulders', 'Arms', 'Hands',
        'Chest', 'Back', 'Abdomen', 'Hips',
        'Thighs', 'Knees', 'Calves', 'Feet', 'Toes'
    ]
    
    fig = go.Figure()
    
    # Create body outline using scatter points
    body_x = [0, -0.2, -0.3, -0.15, -0.3, -0.15, 0, 0.15, 0.3, 0.15, 0.3, 0.2, 0]
    body_y = [1, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    
    fig.add_trace(go.Scatter(
        x=body_x,
        y=body_y,
        fill='toself',
        fillcolor='rgba(100, 181, 246, 0.2)',
        line=dict(color='#64B5F6', width=2),
        hoverinfo='skip'
    ))
    
    # Add body part labels with animations
    for i, part in enumerate(body_parts):
        y_pos = 1 - (i / len(body_parts))
        fig.add_trace(go.Scatter(
            x=[0],
            y=[y_pos],
            mode='markers+text',
            name=part,
            text=[part],
            textposition='middle right',
            marker=dict(
                size=10,
                color='#FF69B4',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            hovertemplate=f"Focus on your {part}<br>Notice any sensations<extra></extra>"
        ))
    
    fig.update_layout(
        title='Body Scan Guide',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 0.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(t=50, l=20, r=20, b=20),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play Scan',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 1000, 'redraw': True},
                    'fromcurrent': True,
                    'mode': 'immediate',
                }]
            }]
        }]
    )
    
    return fig

def create_meditation_timer(duration_minutes):
    """Create an interactive meditation timer visualization."""
    fig = go.Figure()
    
    # Create circular timer
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.ones_like(theta)
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        line=dict(color='#64B5F6', width=3),
        fill='toself',
        fillcolor='rgba(100, 181, 246, 0.1)'
    ))
    
    # Add timer text in center
    fig.add_annotation(
        text=f"{duration_minutes}:00",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=24, color='white')
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks='', showline=False),
            angularaxis=dict(showticklabels=False, ticks='')
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(t=30, l=30, r=30, b=30)
    )
    
    return fig 