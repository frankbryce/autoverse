function tick(items) {
  const current = {};
  items.forEach(item => {
    current[`${item[0]}_${item[1]}`] = 0;
  })
  const nextGen = [];
  const birthCands = {};
  items.forEach(item => {
    let nbrs = 0;
    [-1, 0, 1].forEach(i => {
      [-1, 0, 1].forEach(j => {
        const key = `${item[0] + i}_${item[1] + j}`
        if (current.hasOwnProperty(key)) {
          nbrs += 1;
        } else {
          if (!birthCands.hasOwnProperty(key)) {
            birthCands[key] = 0;
          }
          birthCands[key] += 1;
        }
      });
    });
    nbrs -= 1;
    if ([2, 3].includes(nbrs)) {  // Survive
      nextGen.push(item);
    }
  });
  const birth = Object.entries(birthCands)
    .filter(e => e[1] === 3)
    .map(e => e[0].split('_').map(e => parseInt(e)));
  return nextGen.concat(birth);
}

pattern = [
  [1, 5],
  [1, 6],
  [2, 5],
  [2, 6],
  [11, 5],
  [11, 6],
  [11, 7],
  [12, 4],
  [12, 8],
  [13, 3],
  [13, 9],
  [14, 3],
  [14, 9],
  [15, 6],
  [16, 4],
  [16, 8],
  [17, 5],
  [17, 6],
  [17, 7],
  [18, 6],
  [21, 3],
  [21, 4],
  [21, 5],
  [22, 3],
  [22, 4],
  [22, 5],
  [23, 2],
  [23, 6],
  [25, 1],
  [25, 2],
  [25, 6],
  [25, 7],
  [35, 3],
  [35, 4],
  [36, 3],
  [36, 4]
]

const cellWidth = 10;
const cellHeight = 10;
const rows = 30;
const cols = 40;
const generations = 600;
const genTime = 0.1;  // Seconds

d3.select('#glfield')
    .attr('width', cellWidth * cols)
    .attr('height', cellHeight * rows);

function update(selection, data) {
  const cells = selection
    .selectAll("rect").data(data);
  cells.exit().remove();
  cells.enter().append("rect")
      .attr('width', cellWidth)
      .attr('height', cellHeight)
    .merge(cells)
      .attr('x', d => d[0] * cellWidth)
      .attr('y', d => d[1] * cellHeight)
      .style("fill", "black");
}

let gen = pattern
const end = generations * genTime * 1000;
const interval = genTime * 1000;
d3.select('#glfield').call(update, gen);
// return
const t = d3.interval(elapsed => {
  if (elapsed > end || !gen.length) t.stop();
  gen = tick(gen);
  d3.select('#glfield').call(update, gen);
}, interval);