//
// Created by Administrator on 2023/7/19.
//

#include "Image_engine.h"

CImage_engine::CImage_engine() {

}
CImage_engine::~CImage_engine() {

};


string CImage_engine::StringAdd(string Path, string addPath) {
    Path.append(addPath);
    return Path;
}

