window.$ = require('jquery')
import {Delaunay} from "d3-delaunay";
var d3 = require('d3');
import {quadraticTesselation} from './quadratic_tessellation.js';


let semantic_pointing = function (points) {
    let last_x = null;
    let last_y = null;
    let WIDTH = 50;

    let get_scale = function (x, y) {
        let scale = 1.0;
        let d = 1000000;
        for(let p of points) {
            let o = Math.sqrt((x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1]));
            if (o < d) {
                d = o;
            }
        }
    };


    return (e) => {
        


        last_x = e.offsetX;
        last_y = e.offsetY;
    };
};

var scores = {

}


window.onload = function () {
    let points = [[50, 50], [50, 100], [130, 50], [100, 100]];
    let bounds = [[25, 25], [25, 150], [150, 25], [150, 150]];
    points = points.concat(bounds);
    let delaunay = Delaunay.from(points);
    let voronoi = delaunay.voronoi([-1000, -1000, 2000, 2000]);

    var base = d3.select("#vis");
    console.log(base);
    var chart = base.append("canvas")
        .attr("width", 900)
        .attr("height", 500);

    var context = chart.node().getContext("2d");

    voronoi.render(context);
    console.log(voronoi);

    window.tess = quadraticTesselation(points, voronoi, 0.5);

    document.getElementById('vis').addEventListener('mousemove', e => {
        console.time('eval time');
        let val = window.tess.eval([e.offsetX, e.offsetY]);
        console.timeEnd('eval time');
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);
        context.beginPath();
        context.arc(e.offsetX, e.offsetY, 2, 0, Math.PI * 2, true);
        context.moveTo(e.offsetX - val.gd[0] / 2 + 2, e.offsetY - val.gd[1] / 2)
        context.arc(e.offsetX - val.gd[0] / 2, e.offsetY - val.gd[1] / 2, 2, 0, Math.PI * 2, true);
        context.strokeStyle = 'rgb(0,0,0)';
        context.stroke();

        context.strokeStyle = 'rgb(255,0,0)';
        for (let v of window.tess.vertices) {
            context.beginPath();
            context.moveTo(v[0] + 2, v[1])
            context.arc(v[0] + 2, v[1], 2, 0, Math.PI * 2, true);
            context.stroke();
        }
        context.strokeStyle = 'rgb(0,255,0)';
        for (let p of points) {
            context.beginPath();
            context.moveTo(p[0] + 2, p[1])
            context.arc(p[0] + 2, p[1], 2, 0, Math.PI * 2, true);
            context.stroke();
        }
    });
}