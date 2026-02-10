---
title: C++ 基础
date: 2025-04-09
categories: [Programming Language]
tags: [C++, Engineering, Notes]
---



# 特殊成员函数

## 拷贝构造、赋值构造与对象切片

在 C++ 中，**拷贝构造函数（Copy Constructor）** 和 **赋值构造函数（Copy Assignment Operator）** 都是用于对象复制的特殊成员函数，但它们的使用场景和语义不同，容易混淆。

✅ 一句话区分：

| 场景 | 调用函数 |
| :--- | :--- |
| 用一个已有对象**初始化一个新对象** | **拷贝构造函数** |
| 用一个已有对象**给另一个已存在的对象赋值** | **赋值构造函数（赋值运算符）** |

### 一、拷贝构造函数（Copy Constructor）

📌 定义格式：

```cpp
ClassName(const ClassName& other);
```

必须用引用传参，不可以用值传递**（防止递归引用）**

📌 触发时机：

- 用一个对象初始化另一个新对象时：

  C++

  ```
  MyClass a;
  MyClass b = a;        // ✅ 拷贝构造
  MyClass c(a);         // ✅ 拷贝构造
  ```

- 函数传参**按值传递**时：

  C++

  ```
  void foo(MyClass obj); // ✅ 调用拷贝构造
  ```

- 函数返回对象**按值返回**时（未优化前）：

  C++

  ```
  MyClass create() {
      MyClass temp;
      return temp; // ✅ 可能调用拷贝构造（NRVO 可能优化掉）
  }
  ```

### 深拷贝与浅拷贝：默认拷贝构造函数与动态成员

当类中包含动态内存（如 `new` 出来的指针）时，编译器生成的默认拷贝构造函数只会做**成员**逐个复制（浅拷贝）：

- 浅拷贝：两个对象的指针指向同一块堆内存，可能导致重复释放、悬空指针。
- 深拷贝：复制堆内存内容，每个对象拥有**独立资源**，生命周期互不影响。

```cpp
#include <cstring>
#include <iostream>
using namespace std;

class Buffer {
private:
    char* data;
    size_t len;

public:
    Buffer(const char* s = "") : len(strlen(s)) {
        data = new char[len + 1];
        memcpy(data, s, len + 1);
    }

    // 深拷贝：复制动态内存
    Buffer(const Buffer& other) : len(other.len) {
        data = new char[len + 1];
        memcpy(data, other.data, len + 1);
    }

    // 深拷贝赋值：处理自赋值，避免泄漏
    Buffer& operator=(const Buffer& other) {
        if (this == &other) return *this;
        char* newData = new char[other.len + 1];
        memcpy(newData, other.data, other.len + 1);
        delete[] data;
        data = newData;
        len = other.len;
        return *this;
    }

    ~Buffer() {
        delete[] data;
    }

    const char* c_str() const { return data; }
};

int main() {
    Buffer a("hello");
    Buffer b = a; // 调用深拷贝构造
    cout << a.c_str() << " | " << b.c_str() << '\n';
}
```

结论：只要类管理了动态资源，就不要依赖默认拷贝构造；至少实现拷贝构造、赋值运算符和析构函数（Rule of Three）。

### 二、赋值构造函数（Copy Assignment Operator）

📌 定义格式：

C++

```
ClassName& operator=(const ClassName& other);
```

📌 触发时机：

- 用一个已有对象给另一个**已存在的对象**赋值时：

  C++

  ```
  MyClass a;
  MyClass b;
  b = a; // ✅ 调用赋值运算符
  ```

### 三、对比总结表

| **特性** | **拷贝构造函数**              | **赋值运算符**                           |
| -------- | ----------------------------- | ---------------------------------------- |
| 用途     | 初始化新对象                  | 给已有对象赋值                           |
| 形式     | `ClassName(const ClassName&)` | `ClassName& operator=(const ClassName&)` |
| 触发     | 初始化、传值、返回值          | 赋值操作                                 |
| 返回值   | 无（构造函数）                | `*this`（支持链式赋值）                  |

------

### 四、示例代码对比

C++

```C++
#include <iostream>
using namespace std;

class MyClass {
public:
    int value;

    // 构造函数
    MyClass(int v = 0) : value(v) {
        cout << "Constructor\n";
    }

    // 拷贝构造函数
    MyClass(const MyClass& other) : value(other.value) {
        cout << "Copy Constructor\n";
    }

    // 赋值运算符
    MyClass& operator=(const MyClass& other) {
        if (this != &other) {
            value = other.value;
            cout << "Copy Assignment Operator\n";
        }
        return *this;
    }
};

int main() {
    MyClass a(10);     // Constructor
    MyClass b = a;     // Copy Constructor（初始化）
    MyClass c;
    c = a;             // Copy Assignment Operator（赋值）
}
```

✅ 输出：

```
Constructor
Copy Constructor
Constructor
Copy Assignment Operator
```

- 如果你定义了**析构函数、拷贝构造、拷贝赋值**中的一个，**建议三者都定义**（Rule of Three）。
- C++11 起，还有**移动构造函数**和**移动赋值运算符**（Rule of Five）。


## 对象切片 (Object Slicing)

**拷贝构造/赋值构造与多态之间**并不是“协同工作”的关系，而是**存在潜在冲突**的。

**它们默认都是“非虚”的，一旦用于多态场景，就会把对象切成“基类子对象”，从而丢失派生类信息——这几乎总是 bug**。

[深入理解C++对象切片（Object Slicing）：从 benign bug 到 dangerous corruption](https://juejin.cn/post/7547891478565011508)

C++ Primer 第5版 15.2.3 ……在对象之间不存在类型转换 page.535

**TODO**

> 关于对象切片: 根本原因就是C++的默认拷贝/赋值/移动 构造函数是**非虚**的，是不支持多态的，是静态绑定而非运行时动态绑定的。
>
> 始终牢记C++默认采用静态绑定和非虚赋值操作，并通过使用指针、引用、智能指针和谨慎的类设计来规避这一陷阱。

### 警惕标准库容器

将派生类对象直接存入`std::vector<Base>`会发生**切片**。

正确做法是使用`std::vector<std::unique_ptr<Base>>`。

------

# 多态与继承

### why override

`override` 是 C++11 引入的一个上下文关键字，用于**显式地**指出一个成员函数是重写基类的虚函数。它的作用是**帮助编译器检查是否真的重写了基类的虚函数**，从而避免因为拼写错误、参数列表不一致等问题导致的“**隐藏**”而非“**重写**”的情况。

Simple as that! 



**override关键字不可以用于重写基类的普通成员函数**。

在 C++ 中，`override` **只能用于重写基类的虚函数（`virtual`）**。  
如果基类中的成员函数**不是虚函数**，那么在派生类中使用 `override` 会导致编译错误：

> error: 'void Derived::func()' marked 'override', but does not override any base class function

---

✅ 正确示例：基类函数是虚函数

```cpp
class Base {
public:
    virtual void func();  // 虚函数
};

class Derived : public Base {
public:
    void func() override;  // ✅ 正确，重写虚函数
};
```

---

❌ 错误示例：基类函数不是虚函数

```cpp
class Base {
public:
    void func();  // 普通成员函数，不是虚函数
};

class Derived : public Base {
public:
    void func() override;  // ❌ 错误：override 只能用于虚函数
};
```

---

总结

| 基类函数是否为虚函数 | 是否可以使用 `override` |
| -------------------- | ----------------------- |
| 是                   | ✅ 可以                  |
| 否                   | ❌ 不可以，编译错误      |

---

如果你想“重写”一个**非虚函数**，那实际上只是**隐藏（name hiding）**，不是多态意义上的重写，也不能使用 `override`。  
如果你需要多态行为，必须把基类函数声明为 `virtual`。

## 构造与析构，继承与多态

### 1. 构造顺序

构造的顺序永远是从基类到派生类逐级构造，这是语法规则，与多态无关。

> **对象构造的完整流程**：
>
> 1. **分配对象内存**
> 2. **执行父类构造（如果有）**
> 3. **执行成员初始化列表 - 按声明顺序初始化所有成员 - 未在列表中的成员使用默认初始化**
> 4. **执行构造函数体**

#### 初始化列表：

初始化列表并不决定各成员变量的初始化顺序，实际的初始化顺序由成员变量的声明顺序决定。因此初始化列表的顺序最好保持与声明顺序一致。

**初始化≠赋值**：初始化列表是真正的初始化，构造函数体内是赋值。

无论是否有初始化列表、无论成员变量是否在初始化列表中，都会在进入构造函数体前进行初始化。

——**因此**：存在以下四种必须使用初始化列表的情况：`const`成员变量；引用成员变量；没有默认构造函数的类成员；父类构造函数调用。

**在C++中，初始化总是优于赋值**，对于类类型，直接初始化比默认构造+赋值更高效、更快。

> **最佳实践：**
>
> 1. **总是使用初始化列表**
> 2. **保持初始化顺序与声明顺序一致**
> 3. **基本类型也要初始化**

#### **构造顺序是静态规则**。

因为派生类的对象里“包含”一个基类子对象，必须先构造好基类部分，才能继续构造派生类部分。

编译器在派生类构造函数体执行之前自动插入基类构造函数的调用。

编译器会在派生类构造函数的**初始化阶段**（而非构造函数体内部的开头）自动插入基类构造函数的调用代码。

C++ 语言规则强制要求：**派生类对象的构造必须先初始化其继承的基类部分，再初始化派生类自身的成员和构造函数体**。这一规则由编译器通过以下方式在编译时落实：

1. **隐式调用场景**：如果派生类构造函数没有显式指定基类构造函数（即初始化列表中未写基类名），编译器会自动插入**基类的默认构造函数（无参构造）** 调用，且该调用发生在：
   - 派生类自身成员的初始化之前；
   - 派生类构造函数体执行之前。
2. **显式调用场景**：如果派生类构造函数在**初始化列表**中显式指定了基类构造函数（含带参构造），编译器会按指定的构造函数调用，且顺序仍优先于派生类成员初始化和构造函数体。

构造函数的执行分为两个阶段：

- **初始化阶段**：先执行基类构造 → 再执行派生类成员的初始化（按成员声明顺序，而非初始化列表顺序）；
- **函数体阶段**：执行派生类构造函数体内的代码。

编译器的处理是：在**初始化阶段**的最开始，自动生成基类构造函数的调用代码（隐式或显式），而非修改构造函数体。

#### -->> 所以如果基类没有默认构造函数（仅含带参构造），且派生类未显式调用基类带参构造，**编译会直接报错**， 这进一步证明基类构造的调用是编译期强制检查和处理的

### 2. 基类虚析构与多态

首先，**语言规定**：无论析构函数从哪一层开始执行，必须保证所有基类的析构函数也被调用

当基类的析构函数为virtual时，`delete base_ptr`触发**动态绑定**，运行时根据实际的**对象**类型找到对应的析构函数。

即**virtual**决定了从哪一层开始析构。

析构的执行顺序：**派生类析构函数体 → 派生类自身\*成员\*的析构 → 基类析构函数体**

编译派生类析构函数时，编译器会在其 **“清理阶段”**（派生类析构函数体执行完毕后），自动插入基类析构函数的调用代码 —— 无需手动编写 `Base::~Base()`，编译期已确定这一调用逻辑（类似构造函数，但顺序相反）。

而编译器在派生类析构函数末尾**静态插入**的基类析构函数这一语法规则保证整个继承链都被析构。

1. **虚析构的传递性**：基类析构声明为 `virtual` 后，所有派生类的析构都会隐式成为虚析构（无需重复写 `virtual`，C++11 后可加 `override` 明确覆盖）。
2. **禁止手动调用基类析构**：与构造函数不同，手动调用 `Base::~Base()` 会导致基类部分被重复析构（编译器已自动插入一次），引发未定义行为（如内存双重释放）。**（可以编译通过）**
3. **析构函数的访问权限**：基类析构必须是 `public` 或 `protected`（通常为 `public`），否则编译器无法在派生类析构后自动调用，导致编译错误。
4. **无多态场景可不用虚析构**：若不会通过基类指针 / 引用操作派生类对象（仅直接创建派生类对象），基类析构可不为虚，编译器仍会按顺序自动调用。
5. **纯虚析构的注意事项**：若基类是抽象类（含纯虚函数），可声明纯虚析构（`virtual ~Base() = 0`），但必须提供定义（否则链接错误）—— 因为派生类析构后仍需调用基类析构。

#### Why virtual destructor

> 虚函数是C++多态的实现，没有虚函数就没有多态。
>
> 那么基类的析构函数不是虚析构时，在析构子类时就无法实现**多态**，即无法正确地先调用子类析构再调用基类析构，而是直接调用基类析构，导致子类资源未正确释放。
>
> ```C++
> #include <iostream>
> using namespace std;
> 
> class base {
> public:
>     base() {
>         cout << "base constructor" << endl;
>         b = new int[5];
>     }
>     ~base() {
>         cout << "base destructor" << endl;
>         delete[] b;
>     }
> 
> private:
>     int *b;
> };
> 
> class derived : public base {
> public:
>     derived() {
>         cout << "derived constructor" << endl;
>         d = new int[8];
>     }
>     ~derived() {
>         cout << "derived destructor" << endl;
>         delete[] d;
>     }
> 
> private:
>     int *d;
> };
> 
> int main()
> {
>     base *p = new derived;
>     cout << "---" << endl;
>     delete p;
> 
>     return 0;
> }
> ```
>
> 运行结果：
>
> ```
> base constructor
> derived constructor
> ---
> base destructor
> ```
>
> 将基类的析构函数定义为虚函数后
>
> ```C++
> virtual ~base() {
>     cout << "base destructor" << endl;
>     delete[] b;
> }
> ```
>
> 得到正确的运行结果
>
> ```
> base constructor
> derived constructor
> ---
> derived destructor
> base destructor
> ```
>
> ### 关于这个最基础的继承例子
>
> #### 1. 基类析构为非虚函数
>
> ```C++
> int main()
> {
>     derived *p = new derived;
>     cout << "---" << endl;
>     delete = p;
>     
>     return 0;
> }
> ```
>
> 也能得到正确的运行结果
>
> ```
> base constructor
> derived constructor
> ---
> derived destructor
> base destructor
> ```
>
> 但是这就无法体现多态了
>
> C++多态的核心就是**通过基类指针/引用调用虚函数时，运行时根据对象的实际类型（而不是指针/引用类型）来决定调用哪个函数。**
>
> #### 2. derived *p = new base;
>
> 编译不通过

#### 虚析构函数的工程实践

在工程实践中，C++ 的虚析构函数是一条**“只要类可能被继承，就默认应声明为虚”**的防御性编程铁律。它的核心目的不是“为了多态调用”，而是**“为了在用户通过基类指针 delete 派生对象时，能够按构造逆序完整执行整条析构链，避免资源泄漏+未定义行为”**。

**工业级代码套路：**

1. **基类模板**：声明 `virtual ~Base() = default;` 并禁用拷贝（防止切片）。
2. **抽象接口**：声明纯虚析构 `virtual ~ILogger() = 0;` 并在 .cpp 给出空实现。
3. **工程规范**：“凡是设计为被继承的类，立即把析构函数声明为 `virtual`（或纯虚 + 定义），并关闭拷贝；否则就在类注释里写 ‘本类禁止继承’ 并加 `final`。”

### 3. 虚继承，多重继承

首先是**多继承**可能带来的问题（应该尽量避免多继承）：

比如菱形继承，很可能导致命名冲突、冗余数据等问题。

为了解决多继承时的命名冲突和冗余数据问题，C++ 提出了虚继承，使得在派生类中只保留一份间接基类的成员。

在继承方式前面加上 virtual关键字就是虚继承。

虚继承的目的是让某个类做出声明，承诺愿意共享它的基类。其中，这个被共享的基类就称为**虚基类（Virtual Base Class）**。

在这种机制下，不论虚基类在继承体系中出现了多少次，在派生类中都只包含一份虚基类的成员。

C++标准库中的 iostream 类就是一个虚继承的实际应用案例。iostream 从 istream 和 ostream 直接继承而来，而 istream 和 ostream 又都继承自一个共同的名为 base_ios 的类，是典型的菱形继承。此时 istream 和 ostream 必须采用虚继承，否则将导致 iostream 类中保留两份 base_ios 类的成员。

**PS：如果派生类中定义与基类同名的成员变量或函数时，会隐藏基类的同名成员。或者也可以用`::`访问指定类的成员**

## 详说继承与多态

什么是多态？OOP的核心思想就是多态性。

什么是**多态类型**？具有**继承关系的多个类型**就是多态类型，因为我们能使用这些类型的“多种形式”而无需在意它们的差异。引用或指针的静态类型与动态类型不同这一事实正是C++语言支持多态性的根本所在。

## 函数重载

同一**作用域**内的几个函数名字相同但形参列表不同，称之为**重载（overloaded）函数**。而函数的返回类型不参与重载决策——即重载函数的返回值类型可以不同，如果两个同作用域的函数名字与新参列表都相同只有返回值类型不同，则会导致编译错误：函数重复定义——这也可以由自己很自然地想到：返回值类型是在**函数执行后**才被使用，编译器无法仅凭返回值类型来决定调用哪个函数。

### 重载与const形参

**定义**：在同一作用域中，允许存在多个**同名**函数，但它们的**参数列表必须不同**（类型、个数或顺序）。

**核心特征**：

- **作用域相同**：必须在同一个类、命名空间或全局作用域中
- **参数列表不同**：const 修饰与否、参数类型不同、个数不同都算重载
- **返回类型无关**：仅返回类型不同不能构成重载
- **编译时决议**：由编译器根据实参类型选择最佳匹配

todo

### 虚函数的函数重载与隐藏

```C++
#include <iostream>
#include <string>

class A {
public:
    virtual void show(int a) {
        std::cout << "Class A: " << a << std::endl;
    }
    virtual void show(double b) {
        std::cout << "Class A: " << b << std::endl;
    }
    virtual void show(std::string s) {
        std::cout << "Class A: " << s << std::endl;
    }
};

class B : public A {
public:
    virtual void show(int a) override {
        std::cout << "Class B: " << a << std::endl;
    }
};

int main() {
    B b1;
    A* ptr1 = new B();
    b1.show(10);        // Calls B::show(int)
    b1.show(10.5);     // Calls B::show(int) due to name hiding and implicit conversion
    // b1.show("Hello");  // compilation error due to name hiding
    
    ptr1->show(20);    // Calls B::show(int)
    ptr1->show(20.5);  // Calls A::show(double)
    ptr1->show("World"); // Calls A::show(std::string)
    // simply dynamic binding
    
    delete ptr1;
    return 0;
}
```

使用`B b1`对象直接调用`show()`时发生了**隐藏（Hiding）**：派生类重写一个版本会隐藏基类所有同名函数（包括其他重载版本）。

可以使用`using A::show`来显示引入基类重载版本。

而通过基类指针/引用调用：不受名字隐藏影响，能访问所有基类重载版本，未重写的虚函数调用基类实现。

### 关于命名隐藏 name hiding

[c++: 为什么需要名字隐藏机制(c++ Why name hiding)?](https://blog.csdn.net/wangdingqiaoit/article/details/46502399)





## OOP-派生类与继承类之间的类型转换

OOP的核心思想是数据抽象，继承和动态绑定。通过使用数据抽象，我们可以将类的接口与实现分离；使用继承，可以定义相似的类型并对齐相似关系进行建模；**使用动态绑定（运行时绑定），可以在一定程度上忽略相似类型的区别，而以统一的方式使用他们的对象。**——C++ Primer 第5版 15.1 OOP：概述

——这也是为什么要使用基类指针、引用来操作派生类对象



对于一个指向派生类对象的基类指针，当使用该指针调用基类普通成员函数和派生类重写的虚函数时，行为是不同的。

**调用基类的普通成员函数：`p->nf();`**

1. 编译期

   - `nf` 不是虚函数，所以**直接决议地址**；
   - 生成普通调用指令：
     `call Base::nf`            // 地址在编译期就固定了

2. 运行期

   - 没有任何查表动作；
   - 直接把 `p` 作为 `this` 传给 `Base::nf`（`this` 仍然是 `Der*` 的地址，但函数内部把它当成 `Base*` 使用）。

   结果：无论指针实际指向什么派生对象，都**只会执行基类那份代码**，不会也无法触发动态派发。

**调用被重写的虚函数：`p->vf();`**

1. 编译期

   - 看到指针静态类型是 `Base*`，函数 `vf` 是 `virtual`，于是**不直接生成 call 地址**；
   - 生成一条“虚调用指令”：
     `call [ptr->vptr[k]]`      // k 是 `vf` 在 `Base` 虚表里的槽号

2. 运行期
   a. 取对象头地址 `ptr`（即 `Der` 对象首地址）。
   b. 从对象头里读出 **vptr**（这是 `Der` 类在编译期就填好的指针，指向 `Der` 的虚表）。
   c. 按槽号 `k` 取出表项，得到 `&Der::vf`。
   d. 跳转到该地址执行。

   结果：虽然指针是 `Base*`，实际执行的是 `Der::vf`，实现**动态多态**。



多态的实现：**对象里取 vptr → 查类虚表 → 得到最终函数地址**

> **PS:** 指向派生类对象的基类指针访问派生类独有的普通成员函数？——**无法访问**！静态类型(基类指针)决定了我们能使用哪些成员。
>
> 深究一下：为什么可以到正确的访问虚函数呢？——C++ primer 15.6 继承类中的作用域 page.547



## 详说继承

派生类是否必须再次声明基类虚函数即使不重新定义？——不必



B类继承自A类，C类继承自B类；

B类重新定义了A类的虚函数func1()，C类没有重新定义该虚函数；

在C类对象中调用func1()？在指向C类对象的A类指针(A* ptr)中调用func1()？在指向C类对象的B类指针(B* ptr)中调用func1()？

——都调用的是B::func1()。其中在C类对象中调用是静态绑定，其他是动态绑定通过虚表调用



`struct`关键字与`class`关键字定义的类具有不同的默认访问说明符。

类似地，默认派生运算符也由定义派生类所用的关键字来决定。

```cpp
class Base { /* ... */ };
struct D1 : Base { /* ... */ };	// 默认public继承
class D2 : Base { /* ... */ };	// 默认private继承
```

顺带一提：

> 人们常常有一种错觉，认为在使用`struct`关键字和`class`关键字定义的类之间还有更深层次的差别。
>
> 事实上，**唯一**的差别就是默认成员访问说明符及默认派生访问说明符；除此之外，再无其他差别。**period.**





### 派生类的对象

一个派生类对象包含多个组成部分：一个含有派生类自己定义的（非静态）成员的子对象，以及一个与该派生类继承的基类对应的子对象，如果有多个基类，那么这样的子对象也有多个。——**fact**

> **C++标准并没有明确规定派生类的对象在内存中如何分布**。在一个对象中，继承自基类的部分和派生类自定义的部分不一定是连续存储的。（C++ Primer 第5版 15.2.2 page.530）
>
> ——**待确定**

**每个类控制它自己的成员初始化过程。**

派生类构造函数的实参帮助编译器决定选用哪个构造函数来初始化派生类对象的基类部分。

**首先初始化基类的部分，然后按照声明的顺序依次初始化派生类的成员。**

### 一个继承树中构造函数的调用顺序(from C++ primer)：

```cpp
// C++ primer 15.4
class Quote {
public:
    Quote() = default;
    Quote(const std::string &book, double sales_price): 
    		bookNo(book), price(sales_price) { }
    std::string isbn() const { return bookNo; }
    virtual double net_price(std::size_t n) const { return n * price; }
    virtual ~Quote() = default;
private:
    std::string bookNo;
protected:
    double price = 0.0;
};

class Disc_quote : public Quote {
public:
    Disc_quote() = default;
    Disc_quote(const std::string &book, double price, std::size_t qty, double disc): 
    			Quote(book, price), quantity(qty), discount(disc) { }
    double net_price(std::size_t) const = 0;	// 纯虚函数，使 Disc_quote 成为抽象类，不可实例化
protected:
    std::size_t quantity = 0;
    double disc = 0.0;
};

class Bulk_quote : public Disc_quote {
public:
    Bulk_quote() = default;
    Bulk_quote(const std::string &book, double price, std::size_t qty, double disc): 
    			Disc_quote(book, price, qty, disc) { }
    double net_price(std::size_t) const override;	// 仅声明
};

......
    Bulk_quote bulk;
......
```



1. 基类`Quote`的`net_price`为非纯虚函数，但是直接派生类`Disc_quote`将其override为纯虚函数：首先，**合法语法**；其次，基类的实现仍然可用，派生类依然可以通过`Quote::net_price`来调用默认实现；~~这种模式被称为**“接口强化”**，表达**“必须重新考虑”**~~。
2. **每个类各自控制其对象的初始化过程**：即使`Bulk_quote`没有自己的数据成员，它也仍然需要像原来一样提供一个接受四个参数的构造函数，该构造函数将它的实参传递给`Disc_quote`的构造函数（尽管`Disc_quote`是抽象类我们不能直接定义这个类的对象，但是`Disc_quote`的派生类构造函数将会使用`Disc_quote`的构造函数来构建各个派生类对象的`Disc_quote`部分），然后`Disc_quote`的构造函数继续调用`Quote`的构造函数。`Quote`的构造函数首先初始化`bulk`的`bookNo`和`price`成员，当`Quote`的构造函数结束后，开始运行`Disc_quote`的构造函数并初始化`quantity`和`discount`成员，最后运行`Bulk_quote`的构造函数，当然在这里该函数无须执行实际的初始化或其他工作。



### 继承的访问控制符

派生类的作用域嵌套在基类的作用域**之内**。

因此，对于派生类的一个成员（**函数？**）来说，它使用派生类成员的方式与使用基类成员的方式没什么不同。

> 每个类负责定义各自的接口。想要与类的对象交互必须使用该类的接口，即使这个对象是派生类的基类部分。
>
> 因此，即使从语法上来说可以在派生类构造函数体内给它的public或protected基类成员赋值，也最好不要这么做。应该通过调用基类的构造函数来初始化从基类继承而来的成员。

**每个类**分别**控制**着其成员对于派生类来说是否可**访问**。

```cpp
class Base {
public:
	void pub_mem() { }
protected:
	int prot_mem;
private:
	char priv_mem;
};

struct Pub_Derv : public Base {
    int f() { return prot_mem; }	// 正确，派生类可以访问基类protected成员
    char g() { return priv_mem; }	// 错误，无法访问基类private成员
};

struct Prot_Derv : protected Base {
    int f1() const { return prot_mem; }		// 正确，派生类可以访问基类protected成员
    char g1() const { return priv_mem; }	// 错误，无法访问基类private成员
};

struct Priv_Derv : private Base {
    // private不影响派生**类**访问权限——但是会影响派生类的**用户**的访问权限
    int f1() const { return prot_mem; }		// 正确，派生类可以访问基类protected成员
    char g1() const { return priv_mem; }	// 错误，无法访问基类private成员
};

Pub_Derv d1;	// 继承自Base的成员是public的
Priv_Derv d2;	// 继承自Base的成员是private的
Prot_Derv d3;	// 继承自Base的成员是protected的

d1.pub_mem();	// 正确：pub_mem()在派生类中是public的
d1.prot_mem;	// 错误：成员"Base::prot_mem"不可访问——**类的对象，与类的成员的权限区别**
d1.priv_mem;	// 错误：成员"Base::priv_mem"不可访问

d2.pub_men();	// 错误：pub_mem()在派生类中是private的——类"Priv_Derv"没有成员"pub_mem"——**派生访问说明符起到的控制作用**
d2.prot_mem;	// 错误：成员"Base::prot_mem"不可访问
d2.priv_mem;	// 错误：成员"Base::priv_mem"不可访问

d3.pub_mem();	// 错误：pub_mem()在派生类中是protected的——函数"Base::pub_mem(以声明)”不可访问——类Prot_Derv可以访问，但是类的对象不可以
d3.prot_mem;
d3.priv_mem;
```

从实际使用、程序设计上来看，几乎总是使用`public`继承，`protected`继承的使用几乎总是被认为是程序设计上的失误，而`private`继承则在绝大多数情况下可以被**组合(composition)**的方式来优化替代——除了EBO(空基类优化)及少数其他场景。

> 当然EBO空基类优化并不一定需要`private`继承。
>
> ISO C++ 的规定是任何继承方式都可能进行EBO，但是仅对`private/protected`继承提供了明确的优化保证。



### 类型转换的可访问性

C++ Primer page.544

派生类向基类转换的可访问性是C++继承体系的核心规则之一，它决定了**在代码的特定位置，能否将派生类对象/指针/引用转换为基类类型。**

两个关键因素：

1. **代码的位置**：转换发生在类的内部，还是外部。
2. **继承方式**：public，protected还是private。

总结为一句话：

> **在任何代码位置，如果基类的public成员在该位置可访问，那么派生类向基类的转换也可访问；否则不可访问。**



### 继承中的类作用域

派生类的作用域嵌套在基类的作用域之内。

**静态类型**决定了能使用哪些成员。

但是继承中的作用域嵌套是单向不对称的，基类的成员在派生类中是可见的，但反之则不然。

> 所以指向派生类对象的基类指针不能访问派生类独有的成员/成员函数。

和其他作用域一样，派生类也能**重用**定义在其直接基类或间接基类中的名字，此时定义在内层作用域(即派生类)的名字将隐藏定义在外层作用域(即基类)的名字。

> **派生类的成员将隐藏同名的基类成员。**

依然可以通过作用域运算符`::`来使用被隐藏的基类成员。

作用域运算符的作用就是覆盖原有的查找规则，并只是编译器从指定作用域开始查找。



**最佳实践：**除了覆盖继承而来的虚函数之外，派生类最好不要重用其他定义在基类中的名字。



### 函数调用的解析过程：名字查找与继承

理解函数调用的解析过程对于理解C++的继承至关重要，假定我们调用`p->mem()`（或者`obj.mem()`），则依次执行一下4个步骤：

- 首先确定`p`（或者`obj`）的静态类型。因为我们调用的是一个成员，所以该类型必然是类类型。
- 在`p`（或者`obj`）的静态类型对应的类中查找`mem`。如果找不到，则依次在直接基类中不断查找知道到达继承链的顶端。如果找遍了该类及其基类仍然找不到，则编译器将报错。
- 一旦找到了`mem`，就进行常规的类型检查以确认对于当前找到的`mem`，本次调用是否合法。
- 假设调用合法，则编译器将根据调用的是否是**虚函数**而产生不同的代码：
  - 如果`mem`是虚函数且我们是通过引用或指针进行的调用，则编译器产生的代码将在运行时确定到底运行该虚函数的哪个版本，依据是对象的动态类型。
  - 反之，如果`mem`不是虚函数或者我们是**通过对象（而非有引用或指针）进行的调用**，则编译器将产生一个常规函数调用。

#### 名字查找先于类型检查

**非虚函数的重载与隐藏**

声明在内层作用域的函数并不会重载声明在外层作用域的函数——**函数重载OverLoad**的条件之一就是作用域相同：必须在同一个类、命名空间或全局作用域中。因此，定义派生类中的函数也不会重载其基类成员。和其他作用域一样，如果派生类（即内层作用域）的成员与基类（即外层作用域）的某个成员同名，则派生类将在其作用域内**隐藏**该基类成员。即使派生类成员和基类成员的形参列表不一致（好像构成“重载”一样），基类成员也仍然会被隐藏掉。

```cpp
struct Base {
    int memfcn();
};

struct Derived : Base {
    int memfcn(int);
};

Base b;
Derived d;

b.memfcn();			// 调用Base::memfcn
d.memfcn(10);		// 调用Derived::memfcn
d.memfcn();			// 错误！参数列表为空的memfcn()被隐藏了
d.Base::memfcn();	// 正确：调用Base::memfcn
```

**成员函数与成员变量同名也会触发隐藏name hiding哦**

关于第三条错误的调用语句：为了解析这条调用语句，编译器首先在Derived中查找名字memfcn；因为Derived中确实定义了一个名为memfcn的成员，所以查找过程终止。**一旦名字找到，编译器就不再继续查找了。**Derived中的memfcn需要一个int实参，而当前的调用语句无法提供任何实参，所以该调用语句是错误的。

#### 虚函数与作用域

现在可以理解为什么基类与派生类中的虚函数必须有相同的形参列表了。

假如基类与派生类的虚函数接受的参数不同，则我们就无法通过基类的引用或指针调用派生类的虚函数了。

**也参见前方的小节——虚函数的函数重载与隐藏**

派生类中`override`的虚函数如果参数列表与基类的目标虚函数不同，那么是**不能构成**`override`的，而是构成了隐藏。

```cpp
class Base {
public:
    virtual int fcn();
};

class D1 : public Base {
public:
    // D1继承了Base::fcn的定义
    int fcn(int);	// 形参列表与Base::fcn不一致，隐藏了基类的fcn，这个fcn不是虚函数
    virtual void f2();
};

class D2 : public D1 {
public:
    int fcn(int);	// 是一个非虚函数，隐藏了D1::fcn(int)
    int fcn();		// override了Base的虚函数fcn
    void f2();		// override了D1的虚函数f2
};
```

再看这个例子：

```cpp
#include <iostream>
#include <string>

class Base {
public:
    virtual int fcn1() {
        std::cout << "Base::fcn1()" << std::endl;
        return 0;
    }
    virtual int fcn2() {
        std::cout << "Base::fcn2()" << std::endl;
        return 0;
    }
};

class D1 : public Base {
public:
    int fcn1(int a) {
        std::cout << "D1::fcn1(int)" << std::endl;
        return a;
    }
    virtual int fcn2(int a) {
        std::cout << "D1::fcn2(int)" << std::endl;
        return a;
    }
    virtual void f2() {
        std::cout << "D1::f2()" << std::endl;
    }
};

class D2 : public D1 {
public:
    // int fcn1(int) override; // Error: no matching function to override
    int fcn2(int a) override {
        std::cout << "D2::fcn2(int)" << std::endl;
        return a;
    }
    int fcn2() override {
        std::cout << "D2::fcn2()" << std::endl;
        return 0;
    }	// 成功重写了Base::fcn2()，但是无法通过直接基类D1类型的指针调用！
    void f2() override {
        std::cout << "D2::f2()" << std::endl;
    }	
};

int main() {
    D2 d2;
    Base* pBase = &d2;
    D1* pD1 = &d2;

    pBase->fcn1();      // Calls Base::fcn1()
    // pBase->fcn1(10);    // Error: no matching function to call
    pBase->fcn2();      // Calls D2::fcn2()
    // pBase->fcn2(20);    // Error: no matching function to call
    // pBase->f2();        // Error: no matching function to call

    // pD1->fcn1();       // Error: no matching function to call
    pD1->fcn1(10);     // Calls D1::fcn1(int)
    // pD1->fcn2();       // Error: no matching function to call
    pD1->fcn2(20);     // Calls D2::fcn2(int)
    pD1->f2();         // Calls D2::f2()

    return 0;
}
```

> **是的，`D2::fcn2()` 重写了 `Base::fcn2()`。**
>
> 虽然继承链中存在一些复杂性，但根据C++的虚函数重写规则，答案是肯定的。以下是详细分析：
>
> ### 关键原因
>
> 1. **签名匹配**：`Base::fcn2()` 的签名是 `int fcn2()`，而 `D2::fcn2()` 的签名完全相同。
> 2. **虚函数性质**：`Base::fcn2()` 被声明为 `virtual`，因此可以在派生类中被重写。
> 3. **`override` 关键字的作用**  ：`override` 明确告诉编译器"此函数要重写基类的虚函数"，编译器会执行特殊查找来验证这一点。
>
> ### 关于 `D1::fcn2(int)` 的名字隐藏问题
>
> 在 `D1` 中，`virtual int fcn2(int)` **隐藏**了 `Base::fcn2()`。这意味着：
>
> - 通过 `D1` 类型的指针/引用无法直接调用 `Base::fcn2()`
> - 例如：`D1* d1 = new D2(); d1->fcn2();` 会编译错误
>
> **但是**，名字隐藏**不影响** `D2` 使用 `override` 来重写 `Base::fcn2()`。当编译器处理 `D2::fcn2() override` 时，它会：
>
> 1. 在所有基类中查找名为 `fcn2` 的虚函数
> 2. 查找签名完全匹配的版本（不考虑参数不同的隐藏版本）
> 3. 找到 `Base::fcn2()` 并确认重写关系
>
> ### 实际的虚函数表（vtable）布局
>
> 这个继承结构会产生两个独立的虚函数重载：
>
> - `fcn2()` → 重写 `Base::fcn2()`
> - `fcn2(int)` → 重写 `D1::fcn2(int)`
>
> 它们在 vtable 中占据不同的槽位，互不影响。
>
> ### 使用示例
>
> ```cpp
> Base* b = new D2();
> b->fcn2();      // OK! 调用 D2::fcn2()
> 
> D1* d1 = new D2();
> // d1->fcn2();  // 编译错误！D1::fcn2(int) 隐藏了 Base::fcn2()
> d1->fcn2(42);   // OK! 调用 D2::fcn2(int)
> ```
>
> ### 设计建议
>
> 虽然技术上可行，但这种设计容易造成混淆。**推荐做法**：
>
> 1. 在 `D1` 中使用 `using Base::fcn2;` 来避免隐藏
> 2. 或者给不同功能的函数起不同名字
>
> **总结**：`D2::fcn2()` **确实**重写了 `Base::fcn2()`，但中间类 `D1` 的名字隐藏会导致通过 `D1` 访问时的意外行为。

#### 覆盖重载的函数

和其他函数一样，成员函数无论是否是虚函数都能被重载。

派生类可以覆盖重载函数的0个或多个实例。

如果派生类希望所有的重载版本对于它来说都是可见的，那么它就需要覆盖所有的版本，或者一个也不覆盖。

> 同样参考前面的小节：虚函数的函数重载与隐藏





## 虚函数表

基类的虚函数和派生类的对应虚函数一定在虚函数表里槽号相等

虚函数表是一个“**以基类为模板**”的线性数组。

- 编译器先给基类 `Base` 生成一张表，按**声明顺序**把每个虚函数地址依次放进槽 0、1、2…
- 再给派生类 `Derived` 生成一张表，**先逐字复制基类那张表**（保持槽号不变），然后把**被重写的那几项就地覆盖**成 `Derived` 的新地址；新增虚函数往后追加。

反例只在“复杂继承”出现

- **多重继承**：第二个及以后的基类子对象需要额外的 vptr，槽号可能重新从 0 开始，但那属于“另一张表”，与第一个基类无关。
- **虚继承**（virtual base）：虚基类子对象的 vptr 布局由编译器自行约定，槽号可能不同，但此时派生类通常还会生成一个“主 vptr”专门用于最左非虚基类，保证最常用的接口仍保持槽号一致。

### 对象切片： // TODO

向上转换  `dynamci_cast<>`

https://juejin.cn/post/7547891478565011508

## 纯虚函数与抽象类

只要类中有一个纯虚函数，它就是抽象类，不能被实例化。即使是只有析构函数被定义为纯虚函数也是抽象类。（纯析构函数比较特殊，必须提供定义--**可以是空实现**）

```cpp
class Shape {
public:
    virtual void draw() = 0;
    void setColor(const std::string& c) { color = c; }
protected:
    std::string color;
};

class Circle : public Shape {
public:
    void draw() override {
        std::cout << "Drawing Circle\n";
    }
};

void render(Shape* s) {
    s->draw();  // 多态调用
}

int main() {
    Circle c;
    render(&c);  // 多态的实现--输出：Drawing Circle
}
```



------

# 高级语言特性 (RTTI, Template, Keywords)

## RTTI (Run-Time Type Identification)

C++有3个支持RTTI的元素：

1. `dynamic_cast`：安全地将基类指针转换为派生类指针（如果失败返回0/nullptr）。
2. `typeid`：返回对象的类型信息。
3. `type_info`：存储类型信息。

⚠️**RTTI只适用于包含虚函数的类**。*因为只有对于这种类层次结构，才应该将派生类的地址赋给基类指针*。

### dynamic_cast 机制

**交叉转换是否成功，取决于“最派生对象”里到底有没有那个目标子对象；有就调整指针，没有就返回空。**

RTTI 的引入解决了多态层次下安全向下转型的问题，但也带来了性能开销（存储 type_info）和设计上的“味道”（频繁使用 RTTI 可能意味着设计模式有问题）。

### 一个`dynamic_cast`的例子

**现有ABCD四个类，每个类都是同时继承自E和F。现有基类指针pE(E\*)可能指向ABCD中的某一个，当使用dynamic_cast将pE转换为F\*时发生了什么？**

先把对象在内存里的布局想清楚：

```
       +---------+
       |    E    |  <-- 虚表 E
       +---------+
       |    F    |  <-- 虚表 F
       +---------+
```

A、B、C、D 的内存布局都长这样（顺序可能倒过来，但编译器一次编译里对所有派生类保持一致）。  
当你写下

```cpp
E* pE;          // 可能指向 A/B/C/D 的 E 子对象
F* pF = dynamic_cast<F*>(pE);
```

时，真正发生的事情分三步：

1. 运行时通过 `pE` 指向的 **E 子对象的虚表指针** 找到**最派生对象**（A/B/C/D）的 `type_info`。
2. 检查 `type_info` 里有没有一条“我里面还嵌了一个 F 子对象”的记录。  
   因为 A/B/C/D 都同时继承 E 和 F，这条记录必然存在。
3. 编译器根据记录里存放的**字节偏移量**，把 `pE` 的地址加上（或减去）一个固定常数，得到对应 F 子对象的起始地址，返回给你。

所以：

- 只要 `pE` 实际上指向的是 A/B/C/D 的 E 子对象，转换一定成功，返回非空。  
- 如果 `pE` 真的指向一个**独立的对象 E**（不是任何 A/B/C/D 的子对象），运行时找不到“内嵌 F”的记录，`dynamic_cast` 返回 `nullptr`。

一句话：  
**交叉转换是否成功，取决于“最派生对象”里到底有没有那个目标子对象；有就调整指针，没有就返回空。**

### RTTI（dynamic_cast/typeid）的引入原因

- 解决多态层次下安全向下转型；在运行时分辨真实派生类型。
- 提供统一的类型查询接口（`typeid/type_info`），便于调试、日志、框架内部的对象分派。
- 异常匹配与某些框架机制（如插件系统、序列化/反序列化）需要运行时类型识别。

#### 典型缺陷/代价

- 性能与体积：为每个多态类型存储RTTI结构，`dynamic_cast/typeid` 引入常数开销；对热路径可能影响。
- 只能作用于“多态类型”（有虚函数的类）；对POD (plain old data) /无虚函数类无效。
- 设计味道：频繁用 RTTI 往往意味着类型分派逻辑散落，破坏开放封闭原则；更推荐虚函数、访问者、策略等设计。
- 可移植性与配置：有些编译器/构建可能关闭 RTTI（如 -fno-rtti），库边界若 RTTI 信息不一致会导致 `dynamic_cast/typeid` 异常或未定义行为。
- 跨模块/ABI 问题：不同编译单元/共享库若类型不一致或重复定义，`dynamic_cast` 可能失败或 UB。

#### 使用建议

- 在需要安全向下转型或调试信息时用；避免在核心热路径频繁调用。
- 优先用虚函数/接口设计消除分派需求；确需类型分派时集中封装，减少散乱 `dynamic_cast`。

## 模板函数与虚函数

**模板函数可以是虚函数吗？不可以。**

编译会报错：`templates may not be 'virtual'`

**Why:**

本质上来说，模板属于静态多态，而虚函数属于动态多态。

虚函数机制依赖于固定的虚函数表（vtable）结构，而模板成员函数在实例化前无法确定其地址和数量。

**虚函数需要“运行时固定”的 vtable，而模板实例化是“编译期分散”的**，两者在 ABI 层面无法同时满足。

## 常用关键字：static, const, enum class

#### static修饰非成员变量与类的成员变量

**修饰非成员变量：**

内部链接属性，其他文件是否可以访问？具体在哪个C++标准？

**修饰类的成员变量：**

如何初始化？

#### static修饰非成员函数与类的成员函数

**修饰非成员函数：**

在全局作用域中，`static`修饰的函数称为**静态全局函数**，其核心特性是**作用域被限制在当前文件内**（内部链接），其他文件无法访问该函数。其他文件通过`extern`声明也无法调用。多个文件中可以定义同名的`static`函数来避免命名冲突（作用域隔离）。

**修饰类的成员函数：**

在类中，`static`修饰的成员函数称为**静态成员函数**，它属于整个类而非某个对象，是类级别的函数。

一些**本质**：**编译时就已经确定地址**，**无`this`指针**——与多态不沾边了，与虚函数不沾边了。

> **不依赖对象实例。无`this`指针。不能被`virtual`修饰，无多态特性（不依赖对象，无vtable？）。**
>
> 也就是说，定义某个类类型的指针，将该指针赋值为`nullptr`，也仍然可以用该指针调用该类的`static`成员函数——无`this`指针，不依赖对象实例。

#### const修饰成员函数

声明与定义时都需要把`const`放在函数体之前、参数列表之后。

核心作用是**限制该函数对类成员变量的修改**，确保函数不会改变对象的状态（即 “常量成员函数”）。

> **不能修改类的非`mutable`成员变量**
>
> `const`成员函数内部**只能调用其他 const 成员函数**，不能调用非 const 成员函数。
>
> **`const`对象只能调用`const`成员函数。**

### 强类型枚举 enum class

`enum class` 是 C++11 引入的强类型枚举，解决了传统 `enum` 的命名污染和隐式转换问题。

```cpp
enum class Color : uint8_t {
    Red,
    Green,
    Blue
};

enum class Status {
    Ok = 0,
    Warning = 1,
    Error = 2
};
```

关键特性：

- **作用域隔离**：枚举值需要带作用域访问，如 `Color::Red`，不会污染外层命名空间。
- **强类型**：不会自动转换为整数，避免把不同枚举类型混用。
- **可指定底层类型**：如 `: uint8_t`，有利于控制内存和二进制布局。

```cpp
Color c = Color::Red;
// int x = c;                    // 错误：不允许隐式转换
int x = static_cast<int>(c);     // 显式转换才允许

if (c == Color::Green) {
    // ...
}
```

> 建议：在现代 C++ 中优先使用 `enum class`，除非明确需要传统 `enum` 的隐式整数行为。

### lambda 表达式

lambda 表达式也可以用 namespace 匿名空间里的函数或 static 自由函数来替代，但 lambda 更拥抱现代 C++ 变化。

------

# 内存管理与资源控制

## 内存模型：堆与栈

| **特性**       | **栈 (Stack)**  | **堆 (Heap)**                  |
| -------------- | --------------- | ------------------------------ |
| **分配速度**   | 极快 (移动指针) | 较慢 (系统调用, 查找内存块)    |
| **管理方式**   | 自动 (编译器)   | 手动 (new/delete, malloc/free) |
| **空间大小**   | 小 (MB级)       | 大 (GB级)                      |
| **生命周期**   | 作用域内        | 手动控制                       |
| **内存连续性** | 连续            | 可能不连续 (碎片)              |

- **实际开发建议**：优先使用栈。大数据或长生命周期数据用堆，且推荐使用智能指针。

> ### **栈（Stack）**
>
> **典型大小：**
>
> - **Windows**: 默认 1MB（可通过编译器选项 `/STACK` 调整）
> - **Linux**: 默认 8MB（可通过 `ulimit -s` 查看和修改）
> - **嵌入式系统**: 可能只有几十KB
>
> **特点：**
>
> 1. **自动管理**: 由编译器自动分配和释放，无需手动干预
> 2. **速度快**: 内存分配只需移动栈指针，效率极高（接近O(1)）
> 3. **有限空间**: 容量较小，不适合大内存需求
> 4. **连续内存**: 遵循LIFO（后进先出）原则，内存地址连续
> 5. **作用域绑定**: 变量生命周期与作用域严格绑定，离开作用域立即释放
> 6. **易溢出**: 递归过深或局部数组过大易导致栈溢出（Stack Overflow）
>
> **适用场景**: 局部变量、函数参数、小型数组、函数调用帧
>
> ---
>
> ### **堆（Heap）**
>
> **典型大小：**
>
> - **理论上限**: 受进程虚拟地址空间限制（32位系统约4GB，64位系统非常庞大）
> - **实际可用**: 受物理内存+交换分区大小限制
> - **可增长性**: 可动态申请和释放，大小灵活
>
> **特点：**
>
> 1. **手动管理**: 需通过 `new`/`delete` 或 `malloc`/`free` 显式管理
> 2. **速度较慢**: 分配需要查找合适内存块，涉及系统调用，有额外开销
> 3. **空间巨大**: 可利用几乎所有可用系统内存
> 4. **非连续**: 内存块可能分散，通过指针链接
> 5. **灵活生命周期**: 内存从申请到释放期间始终有效，可跨函数传递
> 6. **内存碎片**: 频繁申请释放易产生外部碎片和内部碎片
> 7. **安全性风险**: 管理不当易导致内存泄漏、悬挂指针、重复释放
>
> **适用场景**: 动态数据结构（链表、树）、大型数组、对象生命周期不确定的数据

静态内存用来保存**局部static对象**、**类static数据成员**以及**任何定义在函数之外的变量**。

栈内存用来保存**定义在函数内**的**非static**对象。

分配在静态或栈内存中的对象由编译器**自动创建和销毁**。对于栈对象，仅在其定义的程序块运行时才存在；static对象在程序使用之前分配，**在程序结束时销毁**。

> 在C++程序的内存布局中，**静态内存对应的是“全局/静态存储区”（Static Storage Area）**，这是一个与栈区、堆区并列的独立内存区域。
>
> - C++程序的完整内存结构通常包含：
>   **代码区 → 全局/静态存储区 → 堆区 → 栈区**（由低地址向高地址增长）
> - “静态内存”是逻辑概念（编译期分配、全程存在），其物理载体是**全局/静态存储区**，**既不在栈区也不在堆区**。
> - C++ Primer中“静态内存”的表述是逻辑分类，实际映射到物理内存的“全局/静态存储区”。

除了静态内存和栈内存，每个程序还拥有一个内存池。这部分内存被称为**自由空间 (free store) **或**堆 (heap)**。程序用堆来存储**动态分配 (dynamically allocate)** 的对象——即，那些在程序运行时分配的对象。动态对象的生存期由程序来控制，也就是说，在动态对象不再使用时，我们的代码必须显式地销毁它们。

## RAII (Resource Acquisition Is Initialization)

使用**局部对象**来**管理资源**的技术称为**资源获取即初始化(RAII)**。

这里的资源主要是指操作系统中有限的东西如内存、网络套接字等等，**局部对象是指存储在栈的对象**，它的生命周期是由操作系统来管理的，无需人工介入。

**RAII**充分的利用了C++语言**局部对象自动销毁**的特性来控制资源的生命周期。

于是，很自然联想到，当我们在使用资源的时候，在构造函数中进行初始化，在析构函数中进行销毁。

整个RAII过程总结四个步骤：

1. 设计一个类封装资源

2. 在构造函数中初始化

3. 在析构函数中执行销毁操作

4. 使用时声明一个该对象的类

> **所以说RAII机制是一种对资源申请、释放这种成对的操作的封装，通过这种方式实现在局部作用域内申请资源然后销毁资源。**
>
> 其实是一种根据C++语言基本特性非常自然地想到的资源管理方式。

## 内存对齐 & std::memcpy / std::memmove

[C/C++ 内存对齐详解](https://zhuanlan.zhihu.com/p/30007037)

[从Eigen向量化谈内存对齐](https://zhuanlan.zhihu.com/p/93824687) 向量化运算就是用SSE、AVX等SIMD（Single Instruction Multiple Data）指令集，实现一条指令对多个操作数的运算，从而提高代码的吞吐量，实现加速效果

[C++ 中 new 操作符内幕：new operator、operator new、placement new ](https://www.cnblogs.com/slgkaifa/p/6887887.html)

## new[], delete[]

`new[]` 用于分配对象数组，`delete[]` 用于释放该数组；二者必须成对出现。

```cpp
#include <iostream>
#include <string>
using namespace std;

class Person {
public:
    Person() { cout << "Person()" << endl; }
    ~Person() { cout << "~Person()" << endl; }
    string name;
};

int main()
{
    // 分配 3 个 Person 对象，会调用 3 次构造函数
    Person* p = new Person[3];

    p[0].name = "Tom";
    p[1].name = "Jerry";
    p[2].name = "Alice";

    for (int i = 0; i < 3; ++i) {
        cout << p[i].name << endl;
    }

    // 释放数组，会调用 3 次析构函数（逆序）
    delete[] p;
    p = nullptr;

    return 0;
}
```

注意事项：

1. `new[]` 必须配对 `delete[]`，不能写成 `delete p;`，否则行为未定义。
2. 对于基础类型数组也建议初始化：`int* a = new int[n]{};`。
3. 更推荐使用 `std::vector` 或 `std::unique_ptr<T[]>` 管理动态数组，避免手动释放。

## std::shared_ptr, std::unique_ptr

`std::unique_ptr` 是**独占所有权**，对象只能被一个指针持有，零额外引用计数，开销最小，性能最好。  
`std::shared_ptr` 是**共享所有权**，内部有引用计数（通常是原子操作），使用更方便但有额外性能成本。

```cpp
#include <iostream>
#include <memory>
using namespace std;

struct Resource {
    Resource() { cout << "Resource acquired\n"; }
    ~Resource() { cout << "Resource released\n"; }
    void work() const { cout << "working...\n"; }
};

int main()
{
    // 1) unique_ptr：最快的智能指针，推荐默认优先使用
    unique_ptr<Resource> up = make_unique<Resource>();
    up->work();

    // unique_ptr 不可拷贝，只能移动
    unique_ptr<Resource> up2 = std::move(up);
    if (!up) {
        cout << "up is null after move\n";
    }

    // 2) shared_ptr：多个对象共享同一资源
    shared_ptr<Resource> sp1 = make_shared<Resource>();
    {
        shared_ptr<Resource> sp2 = sp1; // 引用计数 +1
        cout << "use_count = " << sp1.use_count() << endl; // 2
        sp2->work();
    } // sp2 析构，引用计数 -1
    cout << "use_count = " << sp1.use_count() << endl; // 1

    return 0;
}
```

实践建议：

1. 默认使用 `std::unique_ptr`（更轻量、更快）。
2. 只有确实需要共享所有权时再用 `std::shared_ptr`。
3. 优先 `make_unique` / `make_shared`，避免手写 `new`。
4. `shared_ptr` 循环引用会导致内存泄漏，配合 `std::weak_ptr` 打破环。

------

# 多线程

多线程是**不确定的**。



## `std::thread`

[玩转C++11多线程：让你的程序飞起来的std::thread终极指南](https://www.cnblogs.com/xiaokang-coding/p/18889729)

```cpp
void download1()
{
    cout << "开始下载第一个视频..." << endl;
    for (int i = 0; i < 100; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cout << "下载进度:" << i << endl;
    }
    cout << "第一个视频下载完成..." << endl;
}

void download2()
{
    cout << "开始下载第二个视频..." << endl;
    for (int i = 0; i < 100; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        cout << "下载进度:" << i << endl;
    }
    cout << "第二个视频下载完成..." << endl;
}

int main()
{
    cout << "主线程开始运行\n";
    std::thread d2(download2);	// 开启一个执行download2()函数的线程d2
    download1();

    process();
}
```

这个例程有什么可说的？

1. 主线程开启`d2`这个线程执行`download2()`函数来下载第二个视频。什么是主线程？main函数就可以当作主线程。比如在非多线程编程中，程序就只有main函数这一个线程。main函数中的动作就属于主线程的动作。所以在这里就相当于主线程执行`download1()`函数下载第一个视频，子线程`d2`执行`download2()`函数下载第二个视频。
2. 提出第二个问题，在这里如果`download1()`函数耗时更短，即第一个视频先下载完，那么肯定不能这样写代码了，因为第二个视频还没下载完就调用`process()`进行处理当然是不行的。所以需要等`d2`干完活，主线程才能继续去处理两个视频。于是引出`std::thread::join()`。

### `std::thread::join()`

```cpp
void download1()
{
    cout << "开始下载第一个视频..." << endl;
    for (int i = 0; i < 100; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cout << "第一个视频下载进度:" << i << endl;
    }
    cout << "第一个视频下载完成..." << endl;
}

void download2()
{
    cout << "开始下载第二个视频..." << endl;
    for (int i = 0; i < 100; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        cout << "第二个视频下载进度:" << i << endl;
    }
    cout << "第二个视频下载完成..." << endl;
}
void process()
{
    cout << "开始处理两个视频" << endl;
}

int main()
{
    cout << "主线程开始运行\n";
    std::thread d2(download2);
    download1();
    d2.join();
    process();
}
```

**`std::thread::join()`**

> The function returns when the thread execution has completed. This synchronizes the moment this function returns with the completion of all the operations in the thread: This blocks the execution of the thread that calls this function until the function called on construction returns (if it hasn't yet). 

- **谁调用了这个函数？**调用了这个函数的线程对象，一定要等这个线程对象的方法（在构造时传入的方法）执行完毕后（或者理解为这个线程的活干完了），这个`join()`函数才能得到返回。
- **在什么线程环境下调用了这个函数？**上面说了必须要等待线程方法执行完毕后才能返回，那必然是阻塞调用线程的，也就是说如果一个**线程对象**在一个**线程环境**调用了这个函数，那么这个**线程环境**就会被阻塞，直到这个**线程对象在构造时传入的方法**执行完毕后，才能继续往下走，另外如果线程对象在调用`join()`函数之前就已经做完了自己的事情，那么这个函数不会阻塞线程环境，线程环境正常执行。

在上述修改后的例程中，`download2()`函数耗时更久了，明确以上两个问题：

- **谁调用了`join()`函数？**`d2`这个对象调用了`join()`函数，因此必须等待`d2`的下载任务结束了，`d2.join()`这行代码才能得到返回。
- **`d2`在哪个线程环境下调用了`join()`函数？**`d2`在主线程的环境下调用了`join()`函数，因此主线程要等待`d2`的线程工作做完，否则主线程将一直处于block状态；需要注意的是`d2`真正的任务是在另一个线程进行的，但是`d2`调用`join()`函数的动作是在主线程环境下做的。

**Best Practice**:

> **Always join or detach before exit. **

### `std::thread::detach`

当你调用 `thread_obj.detach()` 时，你实际上是在告知 C++ 运行时和操作系统：

1. **分离控制权**：`thread_obj` 这个 `std::thread` 对象将不再拥有底层操作系统线程的控制权。该 `std::thread` 对象可以被销毁而不会影响后台运行的线程。
2. **独立生命周期**：被分离的线程会变成一个独立的、**守护线程（daemon thread）** 。它将继续执行，直到任务完成或者整个进程终止。
3. **自动资源回收**：一旦分离，当这个守护线程执行完毕时，其所占用的系统资源（例如线程栈）会由 C++ 运行时库或操作系统自动回收，你无需手动 `join()` 它来回收资源。

简而言之，`detach()` 的作用就是让线程“自由”地在后台运行，不再需要父线程的显式管理——原来的父线程也无法管理了，detach的线程与原先的主线程的关系变为平等的了，因此无法在原先的主线程中关闭？不过还是可以通过调用操作系统里的功能来强行kill掉。

### C++ 11中的线程对象生命周期

对于C++ 11的`std::thread`，在退出一个线程对象的作用域之前一定需要调用`join()`或`detach()`（当然`join()`更加常见，一般只有守护线程才需要`detach()`方法），这是避免在退出作用域时自动调用线程对象析构函数由于线程对象仍然处于`joinable()`状态而导致`std::terminate()`终止整个程序。

- 也就是相当于说线程对象不能像其他对象一样自动析构，必须显示地“批准”其声明周期结束。

- 这就给人很怪的感觉，这确实破坏了C++ RAII的自动性，是`std::thread`的设计缺陷。

- 实践中应用RAII(`ThreadGuard`或`std::jthread`)来避免手动管理。
- 线程对象**不可复制**，但**可移动**，防止资源重复管理。

```cpp
class ThreadGuard {
    std::thread& t;
public:
    explicit ThreadGuard(std::thread& t_) : t(t_) {}
    ~ThreadGuard() {
        if (t.joinable()) {
            t.join();  // 自动等待，绝不崩溃
        }
    }
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// 使用
void do_work() {
    std::thread t([]{ /* ... */ });
    ThreadGuard guard(t);  // 析构时自动join，无需手动管理
    
    // ... 可能抛出异常
} // 安全退出，自动join
```



### `std::mutex, std::condition_variable`

`std::mutex`互斥锁实践中一般不手动上锁解锁，推荐使用RAII风格的`std::lock_guard, std::unique_lock`

| 特性           | std::lock_guard                                              | std::unique_lock                                             |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 构造时锁定行为 | 必须在构造时锁定互斥量（或接收已锁定的互斥量，用 `std::adopt_lock`），无法延迟锁定 | 支持延迟锁定（用 `std::defer_lock`），构造时不锁定，后续可手动调用 `lock()` 锁定 |
| 手动解锁       | 不支持。锁定后只能在析构时自动解锁，中途无法手动释放         | 支持。可通过 `unlock()` 手动解锁，之后还能通过 `lock()` 重新锁定 |
| 移动/复制      | 不可移动，不可复制（所有权不可转移）                         | 可移动（`std::move`），不可复制（所有权可转移，不能共享）    |
| 超时/尝试锁定  | 不支持。只能阻塞式锁定，无法尝试锁定或设置超时               | 支持。配合 `std::timed_mutex` 等可使用 `try_lock()`、`try_lock_for()`、`try_lock_until()` 等 |
| 条件变量配合   | 不支持。无法与 `std::condition_variable` 配合使用            | 支持。是 `std::condition_variable::wait()` 的强制参数（需临时解锁并重新锁定） |
| 性能开销       | 极低（仅封装基础锁定/解锁逻辑）                              | 略高（因支持更多功能，内部有状态管理）                       |

`std::lock_guard`在构造时锁定互斥锁，在析构时自动解锁，无论是正常退出作用域还是异常退出都能保证互斥锁被释放。避免**死锁**。

`std::condition_variable::wait()` 

谓词：C++中的谓词是**返回布尔值**的**可调用对象**。



### `std::async, std::future`

