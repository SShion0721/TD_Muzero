#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

#include "tdmz/core/action.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace tdmz;

constexpr int kWindowWidth = 1120;
constexpr int kWindowHeight = 760;
constexpr int kBoardLeft = 20;
constexpr int kBoardTop = 20;
constexpr int kCellSize = 56;
constexpr int kBoardPixels = kCellSize * kBoardW;
constexpr int kPanelLeft = kBoardLeft + kBoardPixels + 24;
constexpr UINT_PTR kAutoTimerId = 1;
constexpr UINT kAutoTimerMs = 250;

enum class Tool {
    Inspect,
    BuildBasic,
    BuildSniper,
    BuildAOE,
    BuildSlow,
    Upgrade,
    Sell,
};

struct Button {
    RECT rect{};
    int id = 0;
    const char* label = "";
};

enum ButtonId {
    BtnInspect = 1,
    BtnBasic,
    BtnSniper,
    BtnAOE,
    BtnSlow,
    BtnUpgrade,
    BtnSell,
    BtnWait1,
    BtnWait10,
    BtnAuto,
    BtnReset,
    BtnNextSeed,
    BtnWaveMode,
};

struct AppState {
    std::unique_ptr<TDEngine> engine;
    Tool tool = Tool::Inspect;
    int selected_x = -1;
    int selected_y = -1;
    int seed = 0;
    bool budgeted_waves = false;
    bool auto_run = false;
    float last_reward = 0.0f;
    bool last_done = false;
    std::vector<std::string> log;
};

AppState g_app;

COLORREF tower_color(TowerType type) {
    switch (type) {
        case TowerType::Basic: return RGB(64, 120, 220);
        case TowerType::Sniper: return RGB(135, 70, 190);
        case TowerType::AOE: return RGB(225, 135, 35);
        case TowerType::Slow: return RGB(40, 165, 175);
    }
    return RGB(80, 80, 80);
}

const char* tower_short_name(TowerType type) {
    switch (type) {
        case TowerType::Basic: return "B";
        case TowerType::Sniper: return "N";
        case TowerType::AOE: return "A";
        case TowerType::Slow: return "L";
    }
    return "?";
}

const char* tool_name(Tool tool) {
    switch (tool) {
        case Tool::Inspect: return "Inspect";
        case Tool::BuildBasic: return "Build Basic";
        case Tool::BuildSniper: return "Build Sniper";
        case Tool::BuildAOE: return "Build AOE";
        case Tool::BuildSlow: return "Build Slow";
        case Tool::Upgrade: return "Upgrade";
        case Tool::Sell: return "Sell";
    }
    return "Unknown";
}

void push_log(std::string message) {
    g_app.log.push_back(std::move(message));
    constexpr std::size_t kMaxEntries = 80;
    if (g_app.log.size() > kMaxEntries) {
        const auto remove_count = static_cast<std::ptrdiff_t>(g_app.log.size() - kMaxEntries);
        g_app.log.erase(g_app.log.begin(), g_app.log.begin() + remove_count);
    }
}

void reset_engine(bool increment_seed) {
    if (increment_seed) {
        ++g_app.seed;
    }
    g_app.engine = std::make_unique<TDEngine>(
        kBoardW, kBoardH, static_cast<uint64_t>(g_app.seed), g_app.budgeted_waves);
    g_app.selected_x = -1;
    g_app.selected_y = -1;
    g_app.last_reward = 0.0f;
    g_app.last_done = false;
    g_app.log.clear();

    std::ostringstream out;
    out << "Reset seed=" << g_app.seed
        << " waves=" << (g_app.budgeted_waves ? "budgeted" : "fixed");
    push_log(out.str());
}

void apply_wait(int ticks) {
    if (!g_app.engine) return;
    const StepResult result = g_app.engine->step_wait(ticks);
    g_app.last_reward = result.reward;
    g_app.last_done = result.done;

    std::ostringstream out;
    out << "Wait " << ticks << " tick" << (ticks == 1 ? "" : "s")
        << " reward=" << std::fixed << std::setprecision(2) << result.reward
        << " done=" << (result.done ? "true" : "false");
    push_log(out.str());
}

bool point_in_rect(int x, int y, const RECT& rect) {
    POINT point{x, y};
    return PtInRect(&rect, point) != FALSE;
}

std::vector<Button> make_buttons() {
    std::vector<Button> buttons;
    int y = 180;
    constexpr int width = 138;
    constexpr int height = 30;
    constexpr int gap = 7;
    const int x0 = kPanelLeft;
    const int x1 = kPanelLeft + width + 8;

    auto add = [&](int x, int id, const char* label) {
        buttons.push_back(Button{RECT{x, y, x + width, y + height}, id, label});
    };

    add(x0, BtnInspect, "Inspect [I]");
    add(x1, BtnBasic, "Basic [1]");
    y += height + gap;
    add(x0, BtnSniper, "Sniper [2]");
    add(x1, BtnAOE, "AOE [3]");
    y += height + gap;
    add(x0, BtnSlow, "Slow [4]");
    add(x1, BtnUpgrade, "Upgrade [U]");
    y += height + gap;
    add(x0, BtnSell, "Sell [S]");
    add(x1, BtnWait1, "Wait 1 [W]");
    y += height + gap;
    add(x0, BtnWait10, "Wait 10 [T]");
    add(x1, BtnAuto, g_app.auto_run ? "Stop auto [Space]" : "Auto [Space]");
    y += height + gap;
    add(x0, BtnReset, "Reset [R]");
    add(x1, BtnNextSeed, "Next seed [N]");
    y += height + gap;
    add(x0, BtnWaveMode, g_app.budgeted_waves ? "Waves: Budgeted" : "Waves: Fixed");

    return buttons;
}

Tool tool_for_button(int id) {
    switch (id) {
        case BtnInspect: return Tool::Inspect;
        case BtnBasic: return Tool::BuildBasic;
        case BtnSniper: return Tool::BuildSniper;
        case BtnAOE: return Tool::BuildAOE;
        case BtnSlow: return Tool::BuildSlow;
        case BtnUpgrade: return Tool::Upgrade;
        case BtnSell: return Tool::Sell;
        default: return g_app.tool;
    }
}

bool button_is_selected(int id) {
    switch (id) {
        case BtnInspect: return g_app.tool == Tool::Inspect;
        case BtnBasic: return g_app.tool == Tool::BuildBasic;
        case BtnSniper: return g_app.tool == Tool::BuildSniper;
        case BtnAOE: return g_app.tool == Tool::BuildAOE;
        case BtnSlow: return g_app.tool == Tool::BuildSlow;
        case BtnUpgrade: return g_app.tool == Tool::Upgrade;
        case BtnSell: return g_app.tool == Tool::Sell;
        case BtnAuto: return g_app.auto_run;
        default: return false;
    }
}

void handle_button(HWND window, int id) {
    if (id >= BtnInspect && id <= BtnSell) {
        g_app.tool = tool_for_button(id);
        InvalidateRect(window, nullptr, FALSE);
        return;
    }

    switch (id) {
        case BtnWait1:
            apply_wait(1);
            break;
        case BtnWait10:
            apply_wait(10);
            break;
        case BtnAuto:
            g_app.auto_run = !g_app.auto_run;
            if (g_app.auto_run) {
                SetTimer(window, kAutoTimerId, kAutoTimerMs, nullptr);
                push_log("Auto-run started");
            } else {
                KillTimer(window, kAutoTimerId);
                push_log("Auto-run stopped");
            }
            break;
        case BtnReset:
            reset_engine(false);
            break;
        case BtnNextSeed:
            reset_engine(true);
            break;
        case BtnWaveMode:
            g_app.budgeted_waves = !g_app.budgeted_waves;
            reset_engine(false);
            break;
        default:
            break;
    }
    InvalidateRect(window, nullptr, FALSE);
}

ActionType action_type_for_tool(Tool tool) {
    switch (tool) {
        case Tool::BuildBasic: return ActionType::BuildBasic;
        case Tool::BuildSniper: return ActionType::BuildSniper;
        case Tool::BuildAOE: return ActionType::BuildAOE;
        case Tool::BuildSlow: return ActionType::BuildSlow;
        case Tool::Upgrade: return ActionType::Upgrade;
        case Tool::Sell: return ActionType::Sell;
        case Tool::Inspect: return ActionType::Wait1;
    }
    return ActionType::Wait1;
}

void apply_board_action(int x, int y) {
    g_app.selected_x = x;
    g_app.selected_y = y;

    if (!g_app.engine || g_app.tool == Tool::Inspect) {
        std::ostringstream out;
        out << "Inspect cell (" << x << "," << y << ")";
        push_log(out.str());
        return;
    }

    const Action action{action_type_for_tool(g_app.tool), x, y, 1};
    const int flat_action = encode_action(action);
    const std::vector<uint8_t> legal_mask = g_app.engine->legal_action_mask();
    const bool legal_before = flat_action >= 0 &&
                              flat_action < static_cast<int>(legal_mask.size()) &&
                              legal_mask[static_cast<std::size_t>(flat_action)] != 0;

    const StepResult result = g_app.engine->step_action(flat_action);
    g_app.last_reward = result.reward;
    g_app.last_done = result.done;

    std::ostringstream out;
    out << action_to_string(action)
        << " legal=" << (legal_before ? "true" : "false")
        << " reward=" << std::fixed << std::setprecision(2) << result.reward
        << " done=" << (result.done ? "true" : "false");
    push_log(out.str());
}

void draw_text(HDC dc, int x, int y, const std::string& text) {
    TextOutA(dc, x, y, text.c_str(), static_cast<int>(text.size()));
}

void fill_rect(HDC dc, const RECT& rect, COLORREF color) {
    HBRUSH brush = CreateSolidBrush(color);
    FillRect(dc, &rect, brush);
    DeleteObject(brush);
}

void frame_rect(HDC dc, const RECT& rect, COLORREF color, int width = 1) {
    HPEN pen = CreatePen(PS_SOLID, width, color);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    HGDIOBJ old_brush = SelectObject(dc, GetStockObject(NULL_BRUSH));
    Rectangle(dc, rect.left, rect.top, rect.right, rect.bottom);
    SelectObject(dc, old_brush);
    SelectObject(dc, old_pen);
    DeleteObject(pen);
}

void draw_circle(
    HDC dc,
    int center_x,
    int center_y,
    int radius,
    COLORREF fill,
    COLORREF outline = RGB(30, 30, 30)) {
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = CreatePen(PS_SOLID, 1, outline);
    HGDIOBJ old_brush = SelectObject(dc, brush);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    Ellipse(dc,
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius);
    SelectObject(dc, old_pen);
    SelectObject(dc, old_brush);
    DeleteObject(pen);
    DeleteObject(brush);
}

void draw_board(HDC dc) {
    if (!g_app.engine) return;

    const auto placeable = g_app.engine->compute_placeable_mask();
    for (int y = 0; y < kBoardH; ++y) {
        for (int x = 0; x < kBoardW; ++x) {
            RECT cell{
                kBoardLeft + x * kCellSize,
                kBoardTop + y * kCellSize,
                kBoardLeft + (x + 1) * kCellSize,
                kBoardTop + (y + 1) * kCellSize,
            };

            COLORREF background = RGB(245, 245, 245);
            if ((x + y) % 2 != 0) background = RGB(232, 235, 238);
            if (placeable[y][x]) background = RGB(229, 243, 229);
            if (x == g_app.engine->spawn_x() && y == g_app.engine->spawn_y()) {
                background = RGB(210, 235, 255);
            }
            if (x == g_app.engine->base_x() && y == g_app.engine->base_y()) {
                background = RGB(255, 220, 220);
            }
            if (g_app.engine->grid()[y][x] != 0) {
                background = RGB(220, 220, 220);
            }

            fill_rect(dc, cell, background);
            frame_rect(dc, cell, RGB(150, 150, 150));

            if (x == g_app.engine->spawn_x() && y == g_app.engine->spawn_y()) {
                draw_text(dc, cell.left + 4, cell.top + 3, "SPAWN");
            }
            if (x == g_app.engine->base_x() && y == g_app.engine->base_y()) {
                draw_text(dc, cell.left + 8, cell.top + 3, "BASE");
            }
        }
    }

    if (g_app.selected_x >= 0 && g_app.selected_y >= 0) {
        RECT selected{
            kBoardLeft + g_app.selected_x * kCellSize,
            kBoardTop + g_app.selected_y * kCellSize,
            kBoardLeft + (g_app.selected_x + 1) * kCellSize,
            kBoardTop + (g_app.selected_y + 1) * kCellSize,
        };
        frame_rect(dc, selected, RGB(220, 30, 30), 3);
    }

    for (const Tower& tower : g_app.engine->towers()) {
        const int center_x = kBoardLeft + tower.x * kCellSize + kCellSize / 2;
        const int center_y = kBoardTop + tower.y * kCellSize + kCellSize / 2;
        draw_circle(dc, center_x, center_y, 18, tower_color(tower.type));
        SetTextColor(dc, RGB(255, 255, 255));
        SetBkMode(dc, TRANSPARENT);
        draw_text(dc, center_x - 4, center_y - 8, tower_short_name(tower.type));
        SetTextColor(dc, RGB(20, 20, 20));
        draw_text(dc, center_x + 10, center_y + 6, std::to_string(tower.level));
    }

    for (const Enemy& enemy : g_app.engine->enemies()) {
        const int center_x = kBoardLeft +
                             static_cast<int>(std::lround(enemy.x * kCellSize)) +
                             kCellSize / 2;
        const int center_y = kBoardTop +
                             static_cast<int>(std::lround(enemy.y * kCellSize)) +
                             kCellSize / 2;
        const COLORREF color = enemy.slow_timer > 0.0f
                                   ? RGB(70, 170, 230)
                                   : RGB(205, 60, 60);
        draw_circle(dc, center_x, center_y, 10, color);

        constexpr int bar_width = 28;
        const float hp_ratio = enemy.max_hp > 0.0f
                                   ? std::clamp(enemy.hp / enemy.max_hp, 0.0f, 1.0f)
                                   : 0.0f;
        RECT bar_background{
            center_x - bar_width / 2,
            center_y - 18,
            center_x + bar_width / 2,
            center_y - 14,
        };
        fill_rect(dc, bar_background, RGB(70, 70, 70));
        RECT hp_bar = bar_background;
        hp_bar.right = hp_bar.left + static_cast<int>(bar_width * hp_ratio);
        fill_rect(dc, hp_bar, RGB(70, 210, 80));
    }
}

void draw_panel(HDC dc) {
    if (!g_app.engine) return;

    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(20, 20, 20));

    HFONT title_font = CreateFontA(
        24,
        0,
        0,
        0,
        FW_BOLD,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        "Segoe UI");
    HGDIOBJ old_font = SelectObject(dc, title_font);
    draw_text(dc, kPanelLeft, 20, "TD_MuZero Engine Debug GUI");
    SelectObject(dc, old_font);
    DeleteObject(title_font);

    std::ostringstream stats;
    stats << "Seed: " << g_app.seed
          << "    Mode: " << (g_app.budgeted_waves ? "Budgeted" : "Fixed") << "\n"
          << "Time: " << std::fixed << std::setprecision(0) << g_app.engine->time()
          << "    Wave: " << g_app.engine->wave() << "\n"
          << "Money: " << g_app.engine->money()
          << "    Base HP: " << g_app.engine->base_hp() << "\n"
          << "Enemies: " << g_app.engine->enemies().size()
          << "    Pending: " << g_app.engine->enemies_to_spawn_count() << "\n"
          << "Spawn timer: " << std::setprecision(2) << g_app.engine->spawn_timer()
          << "    Game over: " << (g_app.engine->game_over() ? "YES" : "no") << "\n"
          << "Last reward: " << std::setprecision(2) << g_app.last_reward
          << "    Tool: " << tool_name(g_app.tool);

    std::istringstream stat_lines(stats.str());
    std::string line;
    int y = 58;
    while (std::getline(stat_lines, line)) {
        draw_text(dc, kPanelLeft, y, line);
        y += 20;
    }

    for (const Button& button : make_buttons()) {
        const bool selected = button_is_selected(button.id);
        fill_rect(dc,
                  button.rect,
                  selected ? RGB(190, 220, 250) : RGB(235, 235, 235));
        frame_rect(dc,
                   button.rect,
                   selected ? RGB(40, 100, 190) : RGB(120, 120, 120),
                   selected ? 2 : 1);
        draw_text(dc, button.rect.left + 8, button.rect.top + 7, button.label);
    }

    int info_y = 405;
    draw_text(dc, kPanelLeft, info_y, "Selected cell details");
    info_y += 22;
    if (g_app.selected_x >= 0 && g_app.selected_y >= 0) {
        std::ostringstream selected;
        selected << "Cell: (" << g_app.selected_x << "," << g_app.selected_y << ")";
        draw_text(dc, kPanelLeft, info_y, selected.str());
        info_y += 20;

        bool found_tower = false;
        for (const Tower& tower : g_app.engine->towers()) {
            if (tower.x == g_app.selected_x && tower.y == g_app.selected_y) {
                std::ostringstream tower_info;
                tower_info << "Tower " << tower_short_name(tower.type)
                           << " L" << tower.level
                           << " dmg=" << std::fixed << std::setprecision(1) << tower.damage
                           << " cd=" << tower.cooldown;
                draw_text(dc, kPanelLeft, info_y, tower_info.str());
                info_y += 20;
                found_tower = true;
                break;
            }
        }
        if (!found_tower) {
            draw_text(dc, kPanelLeft, info_y, "Tower: none");
            info_y += 20;
        }

        int enemies_on_cell = 0;
        for (const Enemy& enemy : g_app.engine->enemies()) {
            if (static_cast<int>(std::lround(enemy.x)) == g_app.selected_x &&
                static_cast<int>(std::lround(enemy.y)) == g_app.selected_y) {
                ++enemies_on_cell;
            }
        }
        draw_text(dc,
                  kPanelLeft,
                  info_y,
                  "Enemies on rounded cell: " + std::to_string(enemies_on_cell));
    } else {
        draw_text(dc, kPanelLeft, info_y, "Click a board cell.");
    }

    constexpr int log_top = 500;
    draw_text(dc, kPanelLeft, log_top, "Recent actions/events");
    int log_y = log_top + 22;
    constexpr int max_lines = 10;
    const int start = std::max(0, static_cast<int>(g_app.log.size()) - max_lines);
    for (int index = start; index < static_cast<int>(g_app.log.size()); ++index) {
        std::string entry = g_app.log[static_cast<std::size_t>(index)];
        if (entry.size() > 58) entry.resize(58);
        draw_text(dc, kPanelLeft, log_y, entry);
        log_y += 19;
    }

    draw_text(
        dc,
        20,
        kBoardTop + kBoardPixels + 12,
        "Choose a tool and click a cell. Invalid actions are sent to the engine, receive its penalty, and still advance one tick.");
}

void handle_key(HWND window, WPARAM key) {
    switch (key) {
        case 'I': handle_button(window, BtnInspect); break;
        case '1': handle_button(window, BtnBasic); break;
        case '2': handle_button(window, BtnSniper); break;
        case '3': handle_button(window, BtnAOE); break;
        case '4': handle_button(window, BtnSlow); break;
        case 'U': handle_button(window, BtnUpgrade); break;
        case 'S': handle_button(window, BtnSell); break;
        case 'W': handle_button(window, BtnWait1); break;
        case 'T': handle_button(window, BtnWait10); break;
        case 'R': handle_button(window, BtnReset); break;
        case 'N': handle_button(window, BtnNextSeed); break;
        case VK_SPACE: handle_button(window, BtnAuto); break;
        default: break;
    }
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
        case WM_CREATE:
            reset_engine(false);
            return 0;

        case WM_LBUTTONDOWN: {
            const int mouse_x = GET_X_LPARAM(lparam);
            const int mouse_y = GET_Y_LPARAM(lparam);

            for (const Button& button : make_buttons()) {
                if (point_in_rect(mouse_x, mouse_y, button.rect)) {
                    handle_button(window, button.id);
                    return 0;
                }
            }

            if (mouse_x >= kBoardLeft && mouse_x < kBoardLeft + kBoardPixels &&
                mouse_y >= kBoardTop && mouse_y < kBoardTop + kBoardPixels) {
                const int x = (mouse_x - kBoardLeft) / kCellSize;
                const int y = (mouse_y - kBoardTop) / kCellSize;
                apply_board_action(x, y);
                InvalidateRect(window, nullptr, FALSE);
            }
            return 0;
        }

        case WM_KEYDOWN:
            handle_key(window, wparam);
            return 0;

        case WM_TIMER:
            if (wparam == kAutoTimerId && g_app.auto_run) {
                if (g_app.engine && !g_app.engine->game_over()) {
                    apply_wait(1);
                } else {
                    g_app.auto_run = false;
                    KillTimer(window, kAutoTimerId);
                    push_log("Auto-run stopped at terminal state");
                }
                InvalidateRect(window, nullptr, FALSE);
            }
            return 0;

        case WM_PAINT: {
            PAINTSTRUCT paint{};
            HDC dc = BeginPaint(window, &paint);
            HDC memory_dc = CreateCompatibleDC(dc);
            RECT client{};
            GetClientRect(window, &client);
            HBITMAP bitmap = CreateCompatibleBitmap(dc, client.right, client.bottom);
            HGDIOBJ old_bitmap = SelectObject(memory_dc, bitmap);

            fill_rect(memory_dc, client, RGB(250, 250, 250));
            HFONT font = CreateFontA(
                17,
                0,
                0,
                0,
                FW_NORMAL,
                FALSE,
                FALSE,
                FALSE,
                DEFAULT_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                CLEARTYPE_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Segoe UI");
            HGDIOBJ old_font = SelectObject(memory_dc, font);
            draw_board(memory_dc);
            draw_panel(memory_dc);
            SelectObject(memory_dc, old_font);
            DeleteObject(font);

            BitBlt(dc, 0, 0, client.right, client.bottom, memory_dc, 0, 0, SRCCOPY);
            SelectObject(memory_dc, old_bitmap);
            DeleteObject(bitmap);
            DeleteDC(memory_dc);
            EndPaint(window, &paint);
            return 0;
        }

        case WM_ERASEBKGND:
            return 1;

        case WM_DESTROY:
            KillTimer(window, kAutoTimerId);
            PostQuitMessage(0);
            return 0;

        default:
            return DefWindowProcA(window, message, wparam, lparam);
    }
}

} // namespace

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int show_command) {
    constexpr const char* class_name = "TDMuZeroEngineDebugGui";

    WNDCLASSA window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = class_name;
    window_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
    window_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

    if (!RegisterClassA(&window_class)) {
        MessageBoxA(
            nullptr,
            "Failed to register the debug GUI window class.",
            "TD_MuZero",
            MB_OK | MB_ICONERROR);
        return 1;
    }

    HWND window = CreateWindowExA(
        0,
        class_name,
        "TD_MuZero Engine Debug GUI",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        kWindowWidth,
        kWindowHeight,
        nullptr,
        nullptr,
        instance,
        nullptr);

    if (!window) {
        MessageBoxA(
            nullptr,
            "Failed to create the debug GUI window.",
            "TD_MuZero",
            MB_OK | MB_ICONERROR);
        return 1;
    }

    ShowWindow(window, show_command);
    UpdateWindow(window);

    MSG message{};
    while (GetMessageA(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
    return static_cast<int>(message.wParam);
}

#endif // _WIN32
