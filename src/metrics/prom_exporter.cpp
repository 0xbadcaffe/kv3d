#include "kv3d/metrics/metrics.hpp"

#include <httplib.h>

#include <memory>
#include <string>
#include <thread>

namespace kv3d {

/// Lightweight Prometheus scrape endpoint that runs on a separate port.
/// Used when the main API server is on a different port from the metrics endpoint.
class PromExporter {
public:
    explicit PromExporter(int port = 9090) : port_(port) {}

    void start() {
        thread_ = std::thread([this] {
            srv_.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
                res.set_content(Metrics::instance().prometheus_text(),
                                "text/plain; version=0.0.4");
            });
            srv_.Get("/metrics/csv", [](const httplib::Request&, httplib::Response& res) {
                res.set_content(Metrics::instance().csv_row(), "text/csv");
            });
            srv_.listen("0.0.0.0", port_);
        });
    }

    void stop() {
        srv_.stop();
        if (thread_.joinable()) thread_.join();
    }

    ~PromExporter() { stop(); }

private:
    int port_;
    httplib::Server srv_;
    std::thread thread_;
};

}  // namespace kv3d
