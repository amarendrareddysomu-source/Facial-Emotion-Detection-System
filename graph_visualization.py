import plotly.graph_objects as go
import pandas as pd
import os

def generate_graphs(log_file, output_html="emotions_graph.html"):
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print("Log file not found. Skipping graph generation.")
        return
        
    if df.empty:
        print("Log file is empty. No data to plot.")
        return

    emotion_mapping = {
        'Happy': 6,
        'Surprise': 5,
        'Neutral': 4,
        'Disgust': 3,
        'Angry': 2,
        'Sad': 1,
        'Fear': 0
    }
    
    # Forward-fill if the emotion somehow wasn't mapped, default to Neutral
    df['EmotionValue'] = df['Emotion'].map(emotion_mapping).fillna(4)
    
    # Calculate duration in continuous seconds for smoother plotting
    def time_to_sec(t):
        parts = t.split(':')
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0])*60 + float(parts[1])
        return 0
        
    df['Seconds'] = df['Timestamp'].apply(time_to_sec)
    
    fig = go.Figure()
    
    # Generate traces/graphs for each tracked person independently
    for person_id, group in df.groupby('PersonID'):
        # Ensure chronological order
        group = group.sort_values('Seconds')
        
        # Create a custom data array to be used in hover text
        customdata = group[['Emotion', 'SnapshotPath', 'Timestamp', 'Confidence']].values
        
        hovertemplate = (
            "<b>Person ID:</b> %s<br>" % person_id +
            "<b>Exact Timestamp:</b> %{customdata[2]}<br>" +
            "<b>Emotion:</b> %{customdata[0]} (%{customdata[3]:.1f}%)<br>" +
            "<br>" +
            "<i>Frame Snapshot:</i><br>" +
            "<!-- Plotly natively evaluates HTML inside hover templates --><br>" +
            "<img src='%{customdata[1]}' width='150'><br>" +
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=group['Seconds'],
            y=group['EmotionValue'],
            mode='lines+markers',
            name=f'Person {person_id}',
            customdata=customdata,
            hovertemplate=hovertemplate
        ))

    fig.update_layout(
        title="Emotion Tracking over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Detected Emotion",
        yaxis=dict(
            tickmode='array',
            tickvals=list(emotion_mapping.values()),
            ticktext=list(emotion_mapping.keys())
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
        template="plotly_white"
    )
    
    fig.write_html(output_html)
    print(f"Interactive graph successfully saved to {output_html}")

if __name__ == "__main__":
    # Isolated test block
    if os.path.exists('emotion_log.csv'):
        generate_graphs('emotion_log.csv')
