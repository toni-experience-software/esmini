#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <map>
#include <limits>
#include <cmath>
#include <iomanip>
#include <sstream>

// 1. Define TinyGLTF implementation
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "tiny_gltf.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

// 2. Esmini Headers
#include "RoadManager.hpp"
#include "roadgeom.hpp"

#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/NodeVisitor>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/LineWidth>
#include <osg/Texture2D>
#include <osg/Image>

using namespace tinygltf;

// -----------------------------------------------------------------------------
// Helper: OSG to glTF Data Conversion
// -----------------------------------------------------------------------------

// Grid-based spatial splitting for runtime culling optimization
constexpr float GRID_CELL_SIZE = 100.0f; // 100m x 100m cells

struct GridKey {
    int x, y;

    bool operator<(const GridKey& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }

    static GridKey fromPosition(float x, float y) {
        return GridKey{
            static_cast<int>(std::floor(x / GRID_CELL_SIZE)),
            static_cast<int>(std::floor(y / GRID_CELL_SIZE))
        };
    }
};

struct ExportStats {
    size_t geometries = 0;
    size_t primitives = 0;
    size_t totalVertices = 0;
    size_t skipped = 0;
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();
    double maxZ = std::numeric_limits<double>::lowest();
    std::map<int, size_t> modeCount;
    std::map<std::string, size_t> skipReasonCount;
};

// Visitor to traverse OSG graph and populate glTF model
class OSGToGLTFVisitor : public osg::NodeVisitor {
public:
    OSGToGLTFVisitor(Model* model, osg::Vec3d offset, std::vector<int>* targetChildren, ExportStats* stats)
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN),
          _model(model), _offset(offset), _targetChildren(targetChildren), _stats(stats) {

        // Ensure we visit ALL nodes, ignoring node masks
        setTraversalMask(0xffffffff);
        setNodeMaskOverride(0xffffffff);

        // Create a default material
        Material defaultMat;
        defaultMat.name = "DefaultMaterial";
        defaultMat.pbrMetallicRoughness.baseColorFactor = {0.8, 0.8, 0.8, 1.0};
        defaultMat.pbrMetallicRoughness.metallicFactor = 0.0;
        defaultMat.pbrMetallicRoughness.roughnessFactor = 0.9;
        defaultMat.doubleSided = true;

        // Cache default material
        _defaultMaterialIndex = _model->materials.size();
        _model->materials.push_back(defaultMat);
    }

    ~OSGToGLTFVisitor() {
        // Destructor
    }

    // Interleaved vertex structure for GPU optimization
    struct Vertex {
        float pos[3];
        float norm[3];
        float color[4];
        float uv[2];
        float tangent[4]; // xyz = tangent, w = handedness
    };

    // Structure to accumulate geometry data for True Merging
    struct GeometryBucket {
        int mode; // TINYGLTF_MODE_...
        int materialIndex;
        std::string alphaMode; // "OPAQUE" or "BLEND"
        GridKey gridKey;

        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        // Bounding box for this bucket
        std::vector<double> minVal = {1e30, 1e30, 1e30};
        std::vector<double> maxVal = {-1e30, -1e30, -1e30};
    };

    // Buckets: Map<AlphaMode, Map<MaterialIndex, Map<PrimitiveMode, Map<GridKey, Bucket>>>>
    std::map<std::string, std::map<int, std::map<int, std::map<GridKey, GeometryBucket>>>> _buckets;

    void finalize() {
        std::cout << "[export] Finalizing merged meshes with spatial splitting..." << std::endl;
        int mergedCount = 0;

        for (auto& alphaEntry : _buckets) {
            const std::string& alphaMode = alphaEntry.first;

            for (auto& matEntry : alphaEntry.second) {
                int matIndex = matEntry.first;

                for (auto& modeEntry : matEntry.second) {
                    int mode = modeEntry.first;

                    for (auto& gridEntry : modeEntry.second) {
                        GridKey gridKey = gridEntry.first;
                        GeometryBucket& bucket = gridEntry.second;

                        if (bucket.vertices.empty()) continue;

                        // Check if we can use 16-bit indices
                        size_t vertexCount = bucket.vertices.size();
                        int indexComponentType = (vertexCount < 65535) ?
                            TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT :
                            TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;

                        // Create interleaved vertex buffer
                        int posAcc, normAcc, colAcc, uvAcc, tanAcc, indAcc;
                        createInterleavedAccessors(_model, bucket.vertices, bucket.minVal, bucket.maxVal,
                                                   posAcc, normAcc, colAcc, uvAcc, tanAcc);

                        // Create index accessor with optimized type
                        if (indexComponentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                            std::vector<unsigned short> indices16;
                            indices16.reserve(bucket.indices.size());
                            for (auto idx : bucket.indices) {
                                indices16.push_back(static_cast<unsigned short>(idx));
                            }
                            indAcc = createAccessor(_model, indices16, TINYGLTF_TYPE_SCALAR, indexComponentType);
                        } else {
                            indAcc = createAccessor(_model, bucket.indices, TINYGLTF_TYPE_SCALAR, indexComponentType);
                        }

                        // Create Primitive
                        Primitive prim;
                        prim.mode = mode;
                        prim.attributes["POSITION"] = posAcc;
                        if (normAcc >= 0) prim.attributes["NORMAL"] = normAcc;
                        if (colAcc >= 0) prim.attributes["COLOR_0"] = colAcc;
                        if (uvAcc >= 0) prim.attributes["TEXCOORD_0"] = uvAcc;
                        if (tanAcc >= 0) prim.attributes["TANGENT"] = tanAcc;
                        prim.indices = indAcc;
                        prim.material = matIndex;

                        // Create Mesh (One mesh per bucket)
                        Mesh mesh;
                        mesh.primitives.push_back(prim);

                        // Name it based on material, mode, and grid cell
                        std::string matName = (matIndex >= 0 && matIndex < _model->materials.size()) ?
                            _model->materials[matIndex].name : "Mat";
                        mesh.name = matName + "_" + alphaMode + "_Mode" + std::to_string(mode) +
                                   "_Grid_" + std::to_string(gridKey.x) + "_" + std::to_string(gridKey.y);

                        int meshIdx = _model->meshes.size();
                        _model->meshes.push_back(mesh);

                        // Create Node
                        Node node;
                        node.mesh = meshIdx;
                        node.name = mesh.name;

                        int nodeIdx = _model->nodes.size();
                        _model->nodes.push_back(node);

                        if (_targetChildren) {
                            _targetChildren->push_back(nodeIdx);
                        }
                        mergedCount++;
                    }
                }
            }
        }
        std::cout << "[export] Created " << mergedCount << " spatially-partitioned meshes from "
                  << _stats->geometries << " source geometries." << std::endl;
    }

    void apply(osg::Geode& geode) override {
        for (unsigned int i = 0; i < geode.getNumDrawables(); ++i) {
            osg::Geometry* geom = geode.getDrawable(i)->asGeometry();
            if (geom) {
                processGeometry(geom, &geode);
            }
        }
        traverse(geode);
    }

    void apply(osg::Group& group) override {
        traverse(group);
    }
    
    void apply(osg::Transform& transform) override {
        osg::Matrix matrix;
        transform.computeLocalToWorldMatrix(matrix, this);
        
        pushMatrix(matrix);
        traverse(transform);
        popMatrix();
    }

private:
    Model* _model;
    osg::Vec3d _offset; // Origin offset to subtract from vertices
    std::vector<int>* _targetChildren; // Pointer to the children list of the parent node (e.g. root transform)
    ExportStats* _stats; // Collect stats for debugging
    int _defaultMaterialIndex;
    std::vector<osg::Matrix> _matrixStack;
    std::map<std::string, int> _materialCache; // Cache for materials
    std::map<std::string, int> _imageCache; // Cache for images
    std::map<std::string, int> _textureCache; // Cache for textures
    
    void pushMatrix(const osg::Matrix& matrix) {
        if (_matrixStack.empty()) {
            _matrixStack.push_back(matrix);
        } else {
            _matrixStack.push_back(matrix * _matrixStack.back());
        }
    }

    void popMatrix() {
        if (!_matrixStack.empty()) {
            _matrixStack.pop_back();
        }
    }

    osg::Matrix getCurrentMatrix() const {
        if (_matrixStack.empty()) return osg::Matrix::identity();
        return _matrixStack.back();
    }

    struct MaterialInfo {
        int index;
        bool hasTexture;
        std::string alphaMode; // "OPAQUE" or "BLEND"
        bool hasNormalMap;
    };

    MaterialInfo getOrCreateMaterial(osg::StateSet* ss) {
        if (!ss) {
            // std::cout << "[DEBUG] No StateSet" << std::endl; 
            return {_defaultMaterialIndex, false, "OPAQUE", false};
        }

        osg::Material* mat = dynamic_cast<osg::Material*>(ss->getAttribute(osg::StateAttribute::MATERIAL));
        if (!mat) {
            // std::cout << "[DEBUG] No osg::Material" << std::endl;
            return {_defaultMaterialIndex, false, "OPAQUE", false};
        }

        osg::Vec4 diffuse = mat->getDiffuse(osg::Material::FRONT);
        float alpha = diffuse.a();
        
        std::cout << "[DEBUG] Inspecting Material. Diffuse: (" 
                  << diffuse.r() << ", " << diffuse.g() << ", " << diffuse.b() << ", " << alpha << ")" << std::endl;

        // Detect transparency
        std::string alphaMode = (alpha < 1.0f) ? "BLEND" : "OPAQUE";

        // Texture Check
        osg::Texture2D* tex = dynamic_cast<osg::Texture2D*>(ss->getTextureAttribute(0, osg::StateAttribute::TEXTURE));
        bool hasTexture = false;
        int textureIndex = -1;
        
        if (tex) {
            std::cout << "[DEBUG]   Texture2D attribute found." << std::endl;
            if (tex->getImage()) {
                osg::Image* img = tex->getImage();
                if (img->valid()) {
                    hasTexture = true;
                    std::string imgName = img->getFileName();
                    if (imgName.empty()) {
                        std::stringstream ptrSs;
                        ptrSs << (void*)img;
                        imgName = "embedded_" + ptrSs.str();
                    }
                    std::cout << "[DEBUG]   Image valid. Name: " << imgName 
                              << " Size: " << img->s() << "x" << img->t() 
                              << " Format: " << std::hex << img->getPixelFormat() << std::dec << std::endl;

                    // Check Image Cache
                    int imgIndex = -1;
                    if (_imageCache.find(imgName) != _imageCache.end()) {
                        imgIndex = _imageCache[imgName];
                    } else {
                        // Create Image
                        Image gltfImg;
                        gltfImg.name = imgName;
                        gltfImg.width = img->s();
                        gltfImg.height = img->t();
                        gltfImg.bits = 8;
                        // Pick MIME type from filename extension so the writer knows how to embed
                        std::string ext = imgName;
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext.size() >= 4 && ext.substr(ext.size() - 4) == ".png") {
                            gltfImg.mimeType = "image/png";
                        } else if ((ext.size() >= 4 && ext.substr(ext.size() - 4) == ".jpg") ||
                                   (ext.size() >= 5 && ext.substr(ext.size() - 5) == ".jpeg")) {
                            gltfImg.mimeType = "image/jpeg";
                        } else {
                            // Default to png to keep viewers happy if extension is missing/odd
                            gltfImg.mimeType = "image/png";
                        }
                        gltfImg.uri = imgName; // helpful for non-binary glTF readers
                        
                        GLenum pixelFormat = img->getPixelFormat();
                        int components = 3; // Default to RGB
                        if (pixelFormat == GL_RGBA || pixelFormat == GL_BGRA) {
                            components = 4;
                        } else if (pixelFormat == GL_RGB || pixelFormat == GL_BGR) {
                            components = 3;
                        }
                        // Handle other specific cases or default to 3 components (RGB) if unknown
                        gltfImg.component = components;
                        gltfImg.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;

                        // Copy data
                        size_t dataSize = img->getTotalSizeInBytes();
                        gltfImg.image.resize(dataSize);
                        if (img->data()) {
                            memcpy(gltfImg.image.data(), img->data(), dataSize);
                        }

                        // Handle BGR/BGRA if needed (simple check for common OSG formats)
                        if (pixelFormat == GL_BGRA) {
                            // Swap B and R channels
                            for (size_t i = 0; i < dataSize; i += 4) {
                                 std::swap(gltfImg.image[i], gltfImg.image[i+2]);
                            }
                        } else if (pixelFormat == GL_BGR) {
                             // Swap B and R channels
                            for (size_t i = 0; i < dataSize; i += 3) {
                                 std::swap(gltfImg.image[i], gltfImg.image[i+2]);
                            }
                        }

                        imgIndex = _model->images.size();
                        _model->images.push_back(gltfImg);
                        _imageCache[imgName] = imgIndex;
                    }
                    
                    // Check Texture Cache
                    std::string texKey = std::to_string(imgIndex);
                    if (_textureCache.find(texKey) != _textureCache.end()) {
                        textureIndex = _textureCache[texKey];
                    } else {
                        Texture gltfTex;
                        gltfTex.source = imgIndex;
                        gltfTex.sampler = -1; // Default sampler
                        
                        textureIndex = _model->textures.size();
                        _model->textures.push_back(gltfTex);
                        _textureCache[texKey] = textureIndex;
                    }
                } else {
                     std::cout << "[DEBUG]   Image invalid!" << std::endl;
                }
            } else {
                std::cout << "[DEBUG]   Texture has no image!" << std::endl;
            }
        }
        
        bool hasNormalMap = false; // For tangent generation

        // Material Caching: Include alpha mode and texture index in key
        std::stringstream ssKey;
        ssKey << std::fixed << std::setprecision(2)
              << diffuse.r() << "_" << diffuse.g() << "_" << diffuse.b() << "_" << alpha
              << "_" << (hasTexture ? "1" : "0") << "_" << alphaMode << "_" << textureIndex;
        std::string key = ssKey.str();

        if (_materialCache.find(key) != _materialCache.end()) {
            return {_materialCache[key], hasTexture, alphaMode, hasNormalMap};
        }

        Material glMat;
        if (textureIndex >= 0) {
            // If a texture is present, set baseColorFactor to white to avoid tinting the texture
            glMat.pbrMetallicRoughness.baseColorFactor = {1.0, 1.0, 1.0, alpha};
        } else {
            glMat.pbrMetallicRoughness.baseColorFactor = {diffuse.r(), diffuse.g(), diffuse.b(), alpha};
        }
        glMat.doubleSided = true;
        glMat.alphaMode = alphaMode;
        
        if (textureIndex >= 0) {
            glMat.pbrMetallicRoughness.baseColorTexture.index = textureIndex;
        }

        // Set alpha cutoff for BLEND mode
        if (alphaMode == "BLEND") {
            glMat.alphaCutoff = 0.5;
        }

        // Name material based on cache size (simple index)
        glMat.name = "Mat_" + std::to_string(_model->materials.size()) + "_" + alphaMode;

        int matIdx = _model->materials.size();
        _model->materials.push_back(glMat);
        _materialCache[key] = matIdx;
        return {matIdx, hasTexture, alphaMode, hasNormalMap};
    }

    void processGeometry(osg::Geometry* geom, osg::Node* parentNode) {
        osg::Vec3Array* verts = dynamic_cast<osg::Vec3Array*>(geom->getVertexArray());
        if (!verts || verts->empty()) {
            if (_stats) _stats->skipReasonCount["no_vertices"]++;
            return;
        }

        osg::Vec3Array* norms = dynamic_cast<osg::Vec3Array*>(geom->getNormalArray());
        osg::Vec4Array* cols = dynamic_cast<osg::Vec4Array*>(geom->getColorArray());
        osg::Vec2Array* uvs = dynamic_cast<osg::Vec2Array*>(geom->getTexCoordArray(0));

        // Early Material Lookup
        osg::StateSet* ss = geom->getStateSet();
        if (!ss && parentNode) ss = parentNode->getStateSet();
        MaterialInfo matInfo = getOrCreateMaterial(ss);

        // Debug geometry info
        // std::cout << "[DEBUG] Processing Geometry. Material Idx: " << matInfo.index 
        //           << " HasTexture: " << matInfo.hasTexture 
        //           << " Verts: " << verts->size() << std::endl;
        
        if (cols && !cols->empty()) {
            osg::Vec4 c = (*cols)[0];
            std::cout << "[DEBUG]   Vertex Color (First): (" 
                      << c.r() << ", " << c.g() << ", " << c.b() << ", " << c.a() << ")" << std::endl;
        } else {
            std::cout << "[DEBUG]   No Vertex Colors found." << std::endl;
        }

        osg::Matrix currentMat = getCurrentMatrix();
        bool hasTransform = !_matrixStack.empty();
        osg::Matrix inverseMat;
        if (hasTransform) inverseMat.invert(currentMat);

        // Pre-transform vertices into interleaved Vertex structures
        std::vector<Vertex> localVertices;
        localVertices.reserve(verts->size());

        // Calculate bounding box for grid key
        float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;

        for (unsigned int i = 0; i < verts->size(); ++i) {
            Vertex v;

            // Transform position
            osg::Vec3d pos = (*verts)[i];
            if (hasTransform) pos = pos * currentMat;
            pos = pos - _offset;
            v.pos[0] = static_cast<float>(pos.x());
            v.pos[1] = static_cast<float>(pos.y());
            v.pos[2] = static_cast<float>(pos.z());

            // Update bounds
            if (v.pos[0] < minX) minX = v.pos[0];
            if (v.pos[1] < minY) minY = v.pos[1];
            if (v.pos[0] > maxX) maxX = v.pos[0];
            if (v.pos[1] > maxY) maxY = v.pos[1];

            // Transform normal
            if (norms && i < norms->size()) {
                osg::Vec3 n = (*norms)[i];
                if (hasTransform) n = n * inverseMat;

                double len2 = n.length2();
                if (len2 < 1e-12 || std::isnan(len2)) n.set(0.0f, 0.0f, 1.0f);
                else n.normalize();

                v.norm[0] = n.x();
                v.norm[1] = n.y();
                v.norm[2] = n.z();
            } else {
                v.norm[0] = 0.0f;
                v.norm[1] = 0.0f;
                v.norm[2] = 1.0f;
            }

            // Color
            if (cols && i < cols->size()) {
                osg::Vec4 c = (*cols)[i];
                v.color[0] = c.r();
                v.color[1] = c.g();
                v.color[2] = c.b();
                v.color[3] = c.a();
            } else {
                v.color[0] = 1.0f;
                v.color[1] = 1.0f;
                v.color[2] = 1.0f;
                v.color[3] = 1.0f;
            }

            // UV
            if (uvs && i < uvs->size() && matInfo.hasTexture) {
                osg::Vec2 uv = (*uvs)[i];
                v.uv[0] = uv.x();
                v.uv[1] = uv.y();
            } else {
                v.uv[0] = 0.0f;
                v.uv[1] = 0.0f;
            }

            // Tangent (will be calculated if needed)
            v.tangent[0] = 1.0f;
            v.tangent[1] = 0.0f;
            v.tangent[2] = 0.0f;
            v.tangent[3] = 1.0f; // handedness

            localVertices.push_back(v);
        }

        // Calculate grid key from geometry center
        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;
        GridKey gridKey = GridKey::fromPosition(centerX, centerY);

        // Iterate Primitives and distribute to buckets
        for (unsigned int i = 0; i < geom->getNumPrimitiveSets(); ++i) {
            osg::PrimitiveSet* ps = geom->getPrimitiveSet(i);
            std::vector<unsigned int> indices;
            int mode = TINYGLTF_MODE_TRIANGLES;

            // Convert OSG primitive types to glTF
            if (ps->getMode() == osg::PrimitiveSet::TRIANGLES) {
                mode = TINYGLTF_MODE_TRIANGLES;
                if (const osg::DrawElementsUInt* de = dynamic_cast<const osg::DrawElementsUInt*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (const osg::DrawElementsUShort* de = dynamic_cast<const osg::DrawElementsUShort*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (const osg::DrawElementsUByte* de = dynamic_cast<const osg::DrawElementsUByte*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (dynamic_cast<const osg::DrawArrays*>(ps)) { osg::DrawArrays* da = (osg::DrawArrays*)ps; for(int k=0; k<da->getCount(); ++k) indices.push_back(da->getFirst() + k); }
            } else if (ps->getMode() == osg::PrimitiveSet::TRIANGLE_STRIP) {
                mode = TINYGLTF_MODE_TRIANGLES;
                unsigned int numIndices = ps->getNumIndices();
                for (unsigned int j=0; j < numIndices - 2; ++j) {
                    unsigned int a, b, c;
                    if (j%2==0) { a = ps->index(j); b = ps->index(j+1); c = ps->index(j+2); }
                    else        { a = ps->index(j); b = ps->index(j+2); c = ps->index(j+1); }
                    indices.push_back(a); indices.push_back(b); indices.push_back(c);
                }
            } else if (ps->getMode() == osg::PrimitiveSet::QUADS) {
                mode = TINYGLTF_MODE_TRIANGLES;
                unsigned int numIndices = ps->getNumIndices();
                for (unsigned int j=0; j < numIndices; j+=4) {
                    if (j+3 >= numIndices) break;
                    unsigned int a=ps->index(j), b=ps->index(j+1), c=ps->index(j+2), d=ps->index(j+3);
                    indices.push_back(a); indices.push_back(b); indices.push_back(c);
                    indices.push_back(a); indices.push_back(c); indices.push_back(d);
                }
            } else if (ps->getMode() == osg::PrimitiveSet::QUAD_STRIP) {
                mode = TINYGLTF_MODE_TRIANGLES;
                unsigned int numIndices = ps->getNumIndices();
                for (unsigned int j = 0; j + 3 < numIndices; j += 2) {
                    unsigned int a = ps->index(j), b = ps->index(j + 1), c = ps->index(j + 2), d = ps->index(j + 3);
                    indices.push_back(a); indices.push_back(b); indices.push_back(c);
                    indices.push_back(b); indices.push_back(d); indices.push_back(c);
                }
            } else if (ps->getMode() == osg::PrimitiveSet::POLYGON) {
                mode = TINYGLTF_MODE_TRIANGLES;
                unsigned int numIndices = ps->getNumIndices();
                if (numIndices >= 3) {
                    unsigned int a0 = ps->index(0);
                    for (unsigned int j = 1; j + 1 < numIndices; ++j) {
                        indices.push_back(a0); indices.push_back(ps->index(j)); indices.push_back(ps->index(j + 1));
                    }
                }
            } else if (ps->getMode() == osg::PrimitiveSet::LINES) {
                mode = TINYGLTF_MODE_LINE;
                if (const osg::DrawElementsUInt* de = dynamic_cast<const osg::DrawElementsUInt*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (const osg::DrawElementsUShort* de = dynamic_cast<const osg::DrawElementsUShort*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (const osg::DrawElementsUByte* de = dynamic_cast<const osg::DrawElementsUByte*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (dynamic_cast<const osg::DrawArrays*>(ps)) { osg::DrawArrays* da = (osg::DrawArrays*)ps; for(int k=0; k<da->getCount(); ++k) indices.push_back(da->getFirst() + k); }
            } else if (ps->getMode() == osg::PrimitiveSet::LINE_STRIP) {
                mode = TINYGLTF_MODE_LINE_STRIP;
                if (const osg::DrawElementsUInt* de = dynamic_cast<const osg::DrawElementsUInt*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (const osg::DrawElementsUShort* de = dynamic_cast<const osg::DrawElementsUShort*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (const osg::DrawElementsUByte* de = dynamic_cast<const osg::DrawElementsUByte*>(ps)) for(auto idx : *de) indices.push_back(idx);
                else if (dynamic_cast<const osg::DrawArrays*>(ps)) { osg::DrawArrays* da = (osg::DrawArrays*)ps; for(int k=0; k<da->getCount(); ++k) indices.push_back(da->getFirst() + k); }
            } else {
                if (_stats) _stats->skipped++;
                continue;
            }

            if (indices.empty()) continue;

            // Get bucket with spatial and material splitting
            GeometryBucket& bucket = _buckets[matInfo.alphaMode][matInfo.index][mode][gridKey];
            bucket.materialIndex = matInfo.index;
            bucket.alphaMode = matInfo.alphaMode;
            bucket.mode = mode;
            bucket.gridKey = gridKey;

            // Current vertex offset in the bucket
            unsigned int vertexOffset = bucket.vertices.size();

            // Append vertices
            bucket.vertices.insert(bucket.vertices.end(), localVertices.begin(), localVertices.end());

            // Update Min/Max from local vertices
            for (const auto& v : localVertices) {
                if (v.pos[0] < bucket.minVal[0]) bucket.minVal[0] = v.pos[0];
                if (v.pos[1] < bucket.minVal[1]) bucket.minVal[1] = v.pos[1];
                if (v.pos[2] < bucket.minVal[2]) bucket.minVal[2] = v.pos[2];
                if (v.pos[0] > bucket.maxVal[0]) bucket.maxVal[0] = v.pos[0];
                if (v.pos[1] > bucket.maxVal[1]) bucket.maxVal[1] = v.pos[1];
                if (v.pos[2] > bucket.maxVal[2]) bucket.maxVal[2] = v.pos[2];
            }

            // Append indices (offset by current vertex count in bucket)
            for (unsigned int idx : indices) {
                bucket.indices.push_back(idx + vertexOffset);
            }

            // Calculate tangents for triangles if normal mapping is enabled
            if (matInfo.hasNormalMap && mode == TINYGLTF_MODE_TRIANGLES) {
                calculateTangents(bucket.vertices, bucket.indices, vertexOffset);
            }

            if (_stats) {
                _stats->primitives++;
                _stats->modeCount[mode]++;
            }
        }

        if (_stats) {
            _stats->geometries++;
            _stats->totalVertices += verts->size();
        }
    }

    // Calculate tangent vectors for normal mapping (simplified MikkTSpace-like approach)
    void calculateTangents(std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, unsigned int startIdx) {
        // Process triangles in groups of 3 indices
        for (size_t i = startIdx; i + 2 < indices.size(); i += 3) {
            unsigned int i0 = indices[i];
            unsigned int i1 = indices[i + 1];
            unsigned int i2 = indices[i + 2];

            if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) continue;

            Vertex& v0 = vertices[i0];
            Vertex& v1 = vertices[i1];
            Vertex& v2 = vertices[i2];

            // Calculate edge vectors
            float edge1[3] = {v1.pos[0] - v0.pos[0], v1.pos[1] - v0.pos[1], v1.pos[2] - v0.pos[2]};
            float edge2[3] = {v2.pos[0] - v0.pos[0], v2.pos[1] - v0.pos[1], v2.pos[2] - v0.pos[2]};

            float deltaUV1[2] = {v1.uv[0] - v0.uv[0], v1.uv[1] - v0.uv[1]};
            float deltaUV2[2] = {v2.uv[0] - v0.uv[0], v2.uv[1] - v0.uv[1]};

            float f = deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1];
            if (std::abs(f) < 1e-6f) f = 1.0f;
            f = 1.0f / f;

            float tangent[3];
            tangent[0] = f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]);
            tangent[1] = f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]);
            tangent[2] = f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]);

            // Normalize tangent
            float len = std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
            if (len > 1e-6f) {
                tangent[0] /= len;
                tangent[1] /= len;
                tangent[2] /= len;
            }

            // Apply Gram-Schmidt orthogonalization and store (accumulate for averaging)
            for (auto* v : {&v0, &v1, &v2}) {
                // Gram-Schmidt: t' = normalize(t - n * dot(n, t))
                float dot = v->norm[0] * tangent[0] + v->norm[1] * tangent[1] + v->norm[2] * tangent[2];
                v->tangent[0] += tangent[0] - v->norm[0] * dot;
                v->tangent[1] += tangent[1] - v->norm[1] * dot;
                v->tangent[2] += tangent[2] - v->norm[2] * dot;
            }
        }

        // Normalize accumulated tangents
        for (auto& v : vertices) {
            float len = std::sqrt(v.tangent[0] * v.tangent[0] + v.tangent[1] * v.tangent[1] + v.tangent[2] * v.tangent[2]);
            if (len > 1e-6f) {
                v.tangent[0] /= len;
                v.tangent[1] /= len;
                v.tangent[2] /= len;
            }
            // Handedness already set to 1.0 by default
        }
    }

    // Create interleaved accessors from Vertex array
    void createInterleavedAccessors(Model* model, const std::vector<Vertex>& vertices,
                                    const std::vector<double>& minVal, const std::vector<double>& maxVal,
                                    int& posAcc, int& normAcc, int& colAcc, int& uvAcc, int& tanAcc) {
        if (vertices.empty()) {
            posAcc = normAcc = colAcc = uvAcc = tanAcc = -1;
            return;
        }

        // Check if we have any non-default values
        bool hasNormals = false, hasColors = false, hasUVs = false, hasTangents = false;

        for (const auto& v : vertices) {
            if (v.norm[0] != 0.0f || v.norm[1] != 0.0f || v.norm[2] != 1.0f) hasNormals = true;
            if (v.color[0] != 1.0f || v.color[1] != 1.0f || v.color[2] != 1.0f || v.color[3] != 1.0f) hasColors = true;
            if (v.uv[0] != 0.0f || v.uv[1] != 0.0f) hasUVs = true;
            if (v.tangent[0] != 1.0f || v.tangent[1] != 0.0f || v.tangent[2] != 0.0f) hasTangents = true;
        }

        // Get or create buffer
        int bufferIdx = 0;
        if (model->buffers.empty()) {
            model->buffers.push_back(Buffer());
        }
        Buffer& buffer = model->buffers[bufferIdx];

        // Calculate stride (all attributes present in struct)
        size_t stride = sizeof(Vertex);
        size_t byteLength = vertices.size() * stride;
        size_t byteOffset = buffer.data.size();

        // Align to 4 bytes
        while (byteOffset % 4 != 0) {
            buffer.data.push_back(0);
            byteOffset++;
        }

        // Copy interleaved vertex data
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(vertices.data());
        buffer.data.insert(buffer.data.end(), bytes, bytes + byteLength);

        // Create BufferView
        BufferView bufferView;
        bufferView.buffer = bufferIdx;
        bufferView.byteOffset = byteOffset;
        bufferView.byteLength = byteLength;
        bufferView.byteStride = stride;
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;

        int bufferViewIdx = model->bufferViews.size();
        model->bufferViews.push_back(bufferView);

        // Create accessors with byte offsets into the interleaved buffer
        auto createAttrAccessor = [&](int type, int componentType, size_t offset) -> int {
            Accessor accessor;
            accessor.bufferView = bufferViewIdx;
            accessor.byteOffset = offset;
            accessor.componentType = componentType;
            accessor.count = vertices.size();
            accessor.type = type;
            int idx = model->accessors.size();
            model->accessors.push_back(accessor);
            return idx;
        };

        // Position accessor with min/max
        posAcc = createAttrAccessor(TINYGLTF_TYPE_VEC3, TINYGLTF_COMPONENT_TYPE_FLOAT, offsetof(Vertex, pos));
        model->accessors[posAcc].minValues = minVal;
        model->accessors[posAcc].maxValues = maxVal;

        // Optional attributes
        normAcc = hasNormals ? createAttrAccessor(TINYGLTF_TYPE_VEC3, TINYGLTF_COMPONENT_TYPE_FLOAT, offsetof(Vertex, norm)) : -1;
        colAcc = hasColors ? createAttrAccessor(TINYGLTF_TYPE_VEC4, TINYGLTF_COMPONENT_TYPE_FLOAT, offsetof(Vertex, color)) : -1;
        uvAcc = hasUVs ? createAttrAccessor(TINYGLTF_TYPE_VEC2, TINYGLTF_COMPONENT_TYPE_FLOAT, offsetof(Vertex, uv)) : -1;
        tanAcc = hasTangents ? createAttrAccessor(TINYGLTF_TYPE_VEC4, TINYGLTF_COMPONENT_TYPE_FLOAT, offsetof(Vertex, tangent)) : -1;
    }

    // Templated accessor creator
    template<typename T>
    int createAccessor(Model* model, const std::vector<T>& data, int type, int componentType,
                       const std::vector<double>& minVal = {}, const std::vector<double>& maxVal = {}) {
        if (data.empty()) return -1;

        int bufferIdx = 0;
        if (model->buffers.empty()) {
            model->buffers.push_back(Buffer());
        }
        Buffer& buffer = model->buffers[bufferIdx];

        size_t byteLength = data.size() * sizeof(T);
        size_t byteOffset = buffer.data.size();
        
        // Padding to 4 bytes
        while (byteOffset % 4 != 0) {
            buffer.data.push_back(0);
            byteOffset++;
        }

        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data.data());
        buffer.data.insert(buffer.data.end(), bytes, bytes + byteLength);

        BufferView bufferView;
        bufferView.buffer = bufferIdx;
        bufferView.byteOffset = byteOffset;
        bufferView.byteLength = byteLength;
        if (componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && (type == TINYGLTF_TYPE_VEC3 || type == TINYGLTF_TYPE_VEC4 || type == TINYGLTF_TYPE_VEC2)) {
             bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
             bufferView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
        }
        
        int bufferViewIdx = model->bufferViews.size();
        model->bufferViews.push_back(bufferView);

        Accessor accessor;
        accessor.bufferView = bufferViewIdx;
        accessor.byteOffset = 0;
        accessor.componentType = componentType;
        accessor.count = data.size() / (tinygltf::GetNumComponentsInType(type));
        accessor.type = type;
        
        if (!minVal.empty()) accessor.minValues = minVal;
        if (!maxVal.empty()) accessor.maxValues = maxVal;

        int accessorIdx = model->accessors.size();
        model->accessors.push_back(accessor);

        return accessorIdx;
    }
};


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: export_odr_to_gltf <xodr_file> [output_glb]" << std::endl;
        return 1;
    }

    std::string xodrPath = argv[1];
    std::string outputPath = (argc > 2) ? argv[2] : "output.glb";

    if (!std::filesystem::exists(xodrPath)) {
        std::cerr << "Error: File not found: " << xodrPath << std::endl;
        return 1;
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "OpenDRIVE to glTF Exporter (Optimized)" << std::endl;
    std::cout << "Input:  " << xodrPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // 1. Initialize RoadManager
    std::cout << "Initializing RoadManager..." << std::endl;
    if (!roadmanager::Position::LoadOpenDrive(xodrPath.c_str())) {
        std::cerr << "Failed to load OpenDRIVE file." << std::endl;
        return 1;
    }
    roadmanager::OpenDrive* odr = roadmanager::Position::GetOpenDrive();
    if (!odr) {
        std::cerr << "Failed to get OpenDRIVE instance." << std::endl;
        return 1;
    }
    std::cout << "RoadManager initialized. Roads: " << odr->GetNumOfRoads() << std::endl;

    // 2. Compute Origin (First Geometry Element)
    osg::Vec3d origin(0,0,0);
    {
        // Use the first road's first point as the origin for deterministic reference
        if (odr->GetNumOfRoads() > 0) {
            roadmanager::Road* r = odr->GetRoadByIdx(0);
            if (r) {
                roadmanager::Position pos(r->GetId(), 0.0, 0.0);
                pos.EvaluateZHPR();
                origin.set(pos.GetX(), pos.GetY(), pos.GetZ());
                std::cout << "[export] Using First Geometry Element as Origin: "
                          << origin.x() << ", " << origin.y() << ", " << origin.z() << std::endl;
            } else {
                auto geoOffset = odr->GetGeoOffset();
                origin.set(geoOffset.x_, geoOffset.y_, geoOffset.z_);
            }
        } else {
            auto geoOffset = odr->GetGeoOffset();
            origin.set(geoOffset.x_, geoOffset.y_, geoOffset.z_);
        }
    }

    // 3. Create OSG Root and RoadGeom
    osg::ref_ptr<osg::Group> root = new osg::Group;
    std::string exePath = "/Users/jesper/Downloads/esmini-master"; 

    bool optimize = false; 
    std::cout << "[export] Generating Geometry..." << std::endl;

    roadgeom::RoadGeom* rg = new roadgeom::RoadGeom(
        odr, 
        root, 
        origin, 
        true,  // generate_road_surface
        true,  // generate_road_objects
        false, // add_ground_plane
        exePath, 
        optimize
    );

    if (rg->root_.valid()) {
        bool alreadyAdded = false;
        for(unsigned int i=0; i<root->getNumChildren(); ++i) {
            if (root->getChild(i) == rg->root_.get()) {
                alreadyAdded = true;
                break;
            }
        }
        if (!alreadyAdded) root->addChild(rg->root_);
    }

    if (!root->getNumChildren()) {
        std::cerr << "[export] Warning: OSG Root has no children." << std::endl;
    }

    // 4. Convert to glTF
    Model model;
    model.asset.generator = "esmini-gltf-exporter-optimized";
    model.asset.version = "2.0";
    
    Value::Object extras;
    extras["originX"] = Value(origin.x());
    extras["originY"] = Value(origin.y());
    extras["originZ"] = Value(origin.z());
    model.asset.extras = Value(extras);

    Scene scene;
    scene.name = "DefaultScene";
    model.scenes.push_back(scene);
    model.defaultScene = 0;

    Node rootTransformNode;
    rootTransformNode.name = "SceneRoot_Zup_to_Yup_Transform";
    // Rotate +90 deg around X to go from OSG Z-up to glTF Y-up without flipping
    rootTransformNode.matrix = {
        1, 0, 0, 0,
        0, 0,-1, 0,
        0, 1, 0, 0,
        0, 0, 0, 1
    };
    
    int rootTransformNodeIdx = model.nodes.size();
    model.nodes.push_back(rootTransformNode);
    model.scenes[0].nodes.push_back(rootTransformNodeIdx);

    std::vector<int> childNodes;
    ExportStats stats;
    OSGToGLTFVisitor visitor(&model, osg::Vec3d(0, 0, 0), &childNodes, &stats);
    root->accept(visitor);
    visitor.finalize(); // Important: This generates the merged meshes
    model.nodes[rootTransformNodeIdx].children = std::move(childNodes);

    std::cout << "[export] glTF summary:"
              << " meshes=" << model.meshes.size()
              << " total_source_geometries=" << stats.geometries
              << " total_vertices=" << stats.totalVertices
              << std::endl;

    // 5. Write to File
    TinyGLTF loader;
    std::cout << "Writing " << outputPath << "..." << std::endl;
    
    bool binary = false;
    if (outputPath.size() >= 4 && outputPath.substr(outputPath.size()-4) == ".glb") {
        binary = true;
    }

    std::string err;
    std::string warn;
    bool ret = loader.WriteGltfSceneToFile(&model, outputPath, true, true, true, binary);

    if (!warn.empty()) std::cerr << "Warn: " << warn << std::endl;
    if (!err.empty()) std::cerr << "Err: " << err << std::endl;

    if (!ret) {
        std::cerr << "Failed to save glTF file." << std::endl;
        return 1;
    }

    std::cout << "Success." << std::endl;
    return 0;
}
