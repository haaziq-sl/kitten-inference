#pragma once
// Minimal JSON parser — supports the subset needed for arch JSON files.
// No external dependencies; header-only.
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include "file_open.hpp"

// ──────────────────────────────────────────────────────────────
// JsonValue — variant node for JSON values
// ──────────────────────────────────────────────────────────────
struct JsonValue {
    using ObjectMap = std::unordered_map<std::string, JsonValue>;

    enum Type { Null, Bool, Number, String, Array, Object } type = Null;

    double      number  = 0;
    bool        boolean = false;
    std::string str;
    std::vector<JsonValue> arr;
    std::unique_ptr<ObjectMap> obj;

    JsonValue();
    ~JsonValue();
    JsonValue(const JsonValue& other);
    JsonValue& operator=(const JsonValue& other);
    JsonValue(JsonValue&& other) noexcept;
    JsonValue& operator=(JsonValue&& other) noexcept;

    bool is_null()   const { return type == Null; }
    bool is_bool()   const { return type == Bool; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array()  const { return type == Array; }
    bool is_object() const { return type == Object; }

    int         as_int()    const { return (int)number; }
    float       as_float()  const { return (float)number; }
    double      as_double() const { return number; }
    bool        as_bool()   const { return boolean; }
    const std::string& as_string() const { return str; }

    bool contains(const std::string& key) const {
        return type == Object && obj && obj->count(key) > 0;
    }
    const JsonValue& operator[](const std::string& key) const {
        if (!obj)
            throw std::runtime_error("JSON key not found: " + key);
        auto it = obj->find(key);
        if (it == obj->end())
            throw std::runtime_error("JSON key not found: " + key);
        return it->second;
    }
    JsonValue& operator[](const std::string& key) {
        if (!obj) obj = std::make_unique<ObjectMap>();
        return (*obj)[key];
    }

    size_t size() const {
        if (type == Array)  return arr.size();
        if (type == Object) return obj ? obj->size() : 0;
        return 0;
    }
    const JsonValue& operator[](size_t i) const {
        if (type != Array || i >= arr.size())
            throw std::runtime_error("JSON: array index " + std::to_string(i) + " out of range");
        return arr[i];
    }
    JsonValue& operator[](size_t i) {
        if (type != Array || i >= arr.size())
            throw std::runtime_error("JSON: array index " + std::to_string(i) + " out of range");
        return arr[i];
    }
};

inline JsonValue::JsonValue()
    : obj(std::make_unique<ObjectMap>()) {}

inline JsonValue::~JsonValue() = default;

inline JsonValue::JsonValue(const JsonValue& other)
    : type(other.type),
      number(other.number),
      boolean(other.boolean),
      str(other.str),
      arr(other.arr),
      obj(other.obj ? std::make_unique<ObjectMap>(*other.obj) : nullptr) {}

inline JsonValue& JsonValue::operator=(const JsonValue& other) {
    if (this == &other) return *this;
    type = other.type;
    number = other.number;
    boolean = other.boolean;
    str = other.str;
    arr = other.arr;
    obj = other.obj ? std::make_unique<ObjectMap>(*other.obj) : nullptr;
    return *this;
}

inline JsonValue::JsonValue(JsonValue&& other) noexcept = default;

inline JsonValue& JsonValue::operator=(JsonValue&& other) noexcept = default;

// ──────────────────────────────────────────────────────────────
// Recursive-descent parser
// ──────────────────────────────────────────────────────────────
namespace json_detail {

struct Parser {
    const char* p;
    const char* end;
    // Arrays and objects nest at most this deep: an architecture file comes from the model's repository, and each
    // level costs stack (a few hundred thousand '[' would overflow it).
    static constexpr int kMaxDepth = 64;
    int depth = 0;
    struct Nest {
        int& d;
        explicit Nest(int& depth) : d(depth) {
            if (++d > kMaxDepth) { --d; throw std::runtime_error("JSON: nested too deeply"); }
        }
        ~Nest() { --d; }
    };

    explicit Parser(const char* data, size_t len)
        : p(data), end(data + len) {}

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            ++p;
    }
    char peek()  { skip_ws(); return p < end ? *p : '\0'; }
    char consume() {
        skip_ws();
        return p < end ? *p++ : '\0';
    }
    void expect(char c) {
        char got = consume();
        if (got != c) {
            char msg[64];
            snprintf(msg, sizeof(msg), "JSON: expected '%c', got '%c'", c, got);
            throw std::runtime_error(msg);
        }
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p < end) {
                    switch (*p) {
                        case '"':  s += '"';  break;
                        case '\\': s += '\\'; break;
                        case '/':  s += '/';  break;
                        case 'n':  s += '\n'; break;
                        case 't':  s += '\t'; break;
                        case 'r':  s += '\r'; break;
                        default:   s += *p;   break;
                    }
                }
            } else {
                s += *p;
            }
            ++p;
        }
        if (p >= end)
            throw std::runtime_error("JSON: unterminated string");
        ++p;  // consume closing "
        return s;
    }

    JsonValue parse_number() {
        const char* start = p;
        if (*p == '-') ++p;
        while (p < end && std::isdigit((unsigned char)*p)) ++p;
        if (p < end && *p == '.') {
            ++p;
            while (p < end && std::isdigit((unsigned char)*p)) ++p;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            ++p;
            if (p < end && (*p == '+' || *p == '-')) ++p;
            while (p < end && std::isdigit((unsigned char)*p)) ++p;
        }
        JsonValue v;
        v.type   = JsonValue::Number;
        v.number = std::strtod(start, nullptr);
        return v;
    }

    JsonValue parse_value();

    JsonValue parse_array() {
        Nest nest(depth);
        expect('[');
        JsonValue v; v.type = JsonValue::Array;
        if (peek() == ']') { consume(); return v; }
        v.arr.push_back(parse_value());
        while (peek() == ',') { consume(); v.arr.push_back(parse_value()); }
        expect(']');
        return v;
    }

    JsonValue parse_object() {
        Nest nest(depth);
        expect('{');
        JsonValue v; v.type = JsonValue::Object;
        if (peek() == '}') { consume(); return v; }
        auto key = parse_string();
        expect(':');
        (*v.obj)[key] = parse_value();
        while (peek() == ',') {
            consume();
            auto k = parse_string();
            expect(':');
            (*v.obj)[k] = parse_value();
        }
        expect('}');
        return v;
    }
};

inline JsonValue Parser::parse_value() {
    char c = peek();
    if (c == '"') {
        JsonValue v; v.type = JsonValue::String; v.str = parse_string(); return v;
    }
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == 't') {
        if (p + 4 <= end && p[1]=='r' && p[2]=='u' && p[3]=='e') {
            p += 4;
            JsonValue v; v.type = JsonValue::Bool; v.boolean = true; return v;
        }
    }
    if (c == 'f') {
        if (p + 5 <= end && p[1]=='a' && p[2]=='l' && p[3]=='s' && p[4]=='e') {
            p += 5;
            JsonValue v; v.type = JsonValue::Bool; v.boolean = false; return v;
        }
    }
    if (c == 'n') {
        if (p + 4 <= end && p[1]=='u' && p[2]=='l' && p[3]=='l') {
            p += 4; return JsonValue{};
        }
    }
    if (c == '-' || std::isdigit((unsigned char)c)) return parse_number();
    char msg[64];
    snprintf(msg, sizeof(msg), "JSON: unexpected char '%c'", c);
    throw std::runtime_error(msg);
}

} // namespace json_detail

// ──────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────
inline JsonValue json_parse(const char* data, size_t len) {
    json_detail::Parser p(data, len);
    return p.parse_value();
}

inline JsonValue json_parse(const std::string& s) {
    return json_parse(s.data(), s.size());
}

inline JsonValue json_load(const std::string& path) {
    FILE* f = fopen_utf8(path, "rb");
    if (!f) throw std::runtime_error("Cannot open JSON: " + path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    if (fread(&buf[0], 1, sz, f) != (size_t)sz) {
        fclose(f);
        throw std::runtime_error("Read error: " + path);
    }
    fclose(f);
    return json_parse(buf);
}
