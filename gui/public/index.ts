import { invoke } from "@tauri-apps/api";

import Background from "../assets/background.svg";
import BlackStone from "../assets/black.svg";
import WhiteStone from "../assets/white.svg";

interface ClickResponse {
  board: number[];
  game_status: number;
}

const BOARD_SIZE = 3;

window.onload = () => {
  const root = document.getElementById("root")!;
  const { width, height } = root.getBoundingClientRect();

  const background = (() => {
    const bg = document.createElement("img");
    bg.style.width = `${width}px`;
    bg.style.height = `${height}px`;
    bg.src = Background;
    return bg;
  })();

  const calculateGridCell = getCalculateGridCell(width, height);
  const makeStoneElement = getMakeStoneElement(width, height);

  function sendOnClick(x: number, y: number): void {
    invoke<ClickResponse>("on_click", calculateGridCell(x, y)).then((res) => {
      const { board, game_status } = res;
      root.innerHTML = "";
      root.appendChild(background);

      for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
          const color = board[y * BOARD_SIZE + x];
          if (!isZero(color)) {
            root.appendChild(makeStoneElement(x, y, color));
          }
        }
      }
    });
  }

  sendOnClick(width + 1, height + 1);

  root.onclick = (ev) => {
    sendOnClick(ev.clientX, ev.clientY);
  };
};

function getCalculateGridCell(width: number, height: number) {
  const cellWidth = width / BOARD_SIZE;
  const cellHeight = height / BOARD_SIZE;

  return (x: number, y: number) => {
    return {
      x: Math.floor(x / cellWidth),
      y: Math.floor(y / cellHeight),
    };
  };
}

function getMakeStoneElement(width: number, height: number) {
  const cellWidth = width / BOARD_SIZE;
  const cellHeight = height / BOARD_SIZE;

  return (x: number, y: number, color: number) => {
    const stone = document.createElement("img");
    stone.style.position = "absolute";
    stone.style.left = `${x * cellWidth}px`;
    stone.style.top = `${y * cellHeight}px`;
    stone.style.width = `${cellWidth}px`;
    stone.style.height = `${cellHeight}px`;
    stone.src = color === 1 ? BlackStone : WhiteStone;
    return stone;
  };
}

function isZero(x: number) {
  return Math.abs(x) < 1e-6;
}
