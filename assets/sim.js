window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        update(time, clicks, ticks, fig) {
            let running = clicks % 2;
            if (ctx.triggered_id != 'clock')
                throw 'no update';  // avoid loop
            else if (running)
                throw 'no update';
            let t_max = fig.frames.length - 1;
            if (running && time < t_max)
                time += 1;
            if (time > t_max)
                time = t_max;
            let frame = fig.frames[time];
            fig.data = frame.data;
            fig.layout = frame.layout;
            let text = "Step: {}  Time: {}".format(time, fig['customdata'][time]['time'])
            return [fig, text, ('Pause' if running else 'Play'), time]
    }
});